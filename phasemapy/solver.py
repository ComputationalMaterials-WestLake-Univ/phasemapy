import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
from glob import glob
from monty.json import MontyDecoder, MSONable, MontyEncoder
from pymatgen.core import Element, Composition
from scipy.optimize import curve_fit, minimize_scalar, minimize
from scipy.special._ufuncs import wofz
from scipy.stats import entropy

from phasemapy.parser import ICDDEntry

from phasemapy.dataio import InstanceData

import torch as th
import torch.nn as nn
from torch.nn.functional import conv2d, relu, normalize, softmax
from torch.distributions import Categorical

chemsys = ['Cu', 'Bi', 'V']  #origin ['V', 'Mn', 'Nb']
oxide_system = True
photon_e = 13e3
max_q_shift = 0.02
resample_density = 2000
initial_alphagamma = 0.03
SUM_NORM = 6000

np.set_printoptions(6, suppress=True)


class Autoencoder(nn.Module):
    def __init__(self, basis_xrd, exp_xrd, basis_xrd_weight, basis_comp, sample_comp, shifts, loss_weight):
        self.basis_xrd = basis_xrd
        self.exp_xrd = exp_xrd
        self.sample_density = exp_xrd.shape[1]
        self.basis_xrd_weight = basis_xrd_weight
        self.basis_comp = basis_comp
        self.sample_comp = sample_comp
        self.shifts = shifts
        self.loss_weight = loss_weight
        super(Autoencoder, self).__init__()
        self.N = self.basis_xrd.shape[0]  # N candicate basis
        self.fc1 = nn.Linear(self.sample_density, 40 * self.N)
        self.fc2 = nn.Linear(40 * self.N, 20 * self.N)
        self.fc3 = nn.Linear(20 * self.N, self.N * (self.shifts * 2 + 1))

    def encode(self, x):
        x = self.fc1(x)
        x = relu(x)
        x = self.fc2(x)
        x = relu(x)
        x = self.fc3(x)

        # x = softmax(x,dim=-1)
        x = x.reshape(-1, self.N, self.shifts * 2 + 1)
        x = relu(x)
        x = normalize(x, p=1, dim=(1, 2))
        return x

    def decode(self, x):
        mat1 = self.basis_xrd[None, None, :]
        mat2 = x[:, None, :]
        conv = conv2d(mat1, mat2)
        conv = conv.reshape(-1, self.sample_density)
        return conv

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

    def cal_comp_loss(self):
        x = self.exp_xrd
        encoded = self.encode(x)
        fraction = th.sum(encoded, dim=-1)
        recon_comp = fraction[:, :, None] * self.basis_xrd_weight[None, :, None] * self.basis_comp[None, :, :]
        recon_comp = th.sum(recon_comp, dim=1)
        recon_comp = normalize(recon_comp, p=1)[0]
        return nn.MSELoss(reduction='sum')(recon_comp, self.sample_comp)

    @property
    def weight(self):
        return 1 / (self.exp_xrd + 0.01 * th.max(self.exp_xrd))

    def cal_xrd_loss(self):
        x = self.exp_xrd
        return th.sum(self.weight * (self(x) - x) ** 2) / th.sum(x)
        # return nn.L1Loss(reduction='sum')(self(x), x) / th.sum(x)

    def cal_entropy_loss(self):
        x = self.exp_xrd
        encoded = self.encode(x)
        sample_basis_act = th.sum(encoded, dim=-1)
        print('x' ,sample_basis_act)
        entropy = Categorical(probs=sample_basis_act).entropy()
        print('entropy',entropy,th.mean(entropy))
        return th.mean(entropy) #th.mean(entropy)

    def loss_fn(self):
        xrd_loss = self.cal_xrd_loss()
        comp_loss = self.cal_comp_loss()
        entropy_loss = self.cal_entropy_loss()
        print ('loss',xrd_loss, entropy_loss, comp_loss)
        return xrd_loss, entropy_loss, comp_loss

    def train(self, num_epochs, print_prog=True):
        optimizer = th.optim.Adam(self.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-06, weight_decay=0)
        for epoch in range(num_epochs):
            xrd_loss, entropy_loss, comp_loss = self.loss_fn()
            loss = self.loss_weight['xrd_loss'] * xrd_loss + self.loss_weight['entropy_loss'] * entropy_loss + \
                   self.loss_weight['comp_loss'] * comp_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if epoch % 500 == 0 and print_prog:
                # l1,l2,l3 = xrd_loss.data.cpu(), comp_loss.data.cpu(), entropy_loss.data.cpu()

                print(epoch, loss.data.cpu().numpy(), xrd_loss.data.cpu(), comp_loss.data.cpu(),
                      entropy_loss.data.cpu())
        return

    def plot(self):
        recon = self.forward(self.exp_xrd).cpu().detach().numpy().squeeze()
        plt.plot(np.arange(len(recon)), self.exp_xrd.cpu().numpy().squeeze(), label='orig', alpha=0.5)
        plt.plot(np.arange(len(recon)), recon, label='recon', alpha=0.5)
        plt.legend(loc=1)
        plt.show()

    def solution(self):
        encoded = self.encode(self.exp_xrd).cpu().detach().numpy()[0]
        fractions = np.sum(encoded, axis=1)
        normed_encoded = encoded / np.maximum(1e-6, fractions).reshape(-1, 1)
        shifts = np.sum(np.dot(normed_encoded, np.arange(-self.shifts, self.shifts + 1).reshape(-1, 1)), axis=1)
        stds = np.sqrt(np.mean(np.arange(-self.shifts, self.shifts + 1).reshape(1, -1) ** 2 * normed_encoded, axis=1))

        return fractions, shifts, stds


class Sample(MSONable):
    def __init__(self, sample_id, log_q, exp_xrd, chemsys, composition, oxide_system, wavelength, max_q_shift,
                 solution):
        self.sample_id = sample_id
        self.log_q = log_q
        self.exp_xrd = exp_xrd
        self.chemsys = chemsys
        self.composition = composition
        self.oxide_system = oxide_system
        self.wavelength = wavelength
        self.max_q_shift = max_q_shift
        self.solution = solution

    @property
    def q(self):
        return np.exp(self.log_q)

    @property
    def log_q_spacing(self):
        return self.log_q[1] - self.log_q[0]

    @property
    def default_extension(self):
        return int(np.around(self.max_q_shift / self.log_q_spacing))

    def wider_log_q(self, extension=None):
        if extension is None:
            extension = self.default_extension
        return np.linspace(self.log_q[0] - self.log_q_spacing * extension,
                           self.log_q[-1] + self.log_q_spacing * extension,
                           len(self.log_q) + extension * 2)  # [1:-1]

    @staticmethod
    def get_voigt_xrd(q_vectors, q_loc, intensity, alphagamma, wavelength):
        def Voigt(x, c, alphagamma, amps):
            x = x[:, None]
            c = c[None, :]
            amps = amps[None, :]
            alphagamma = alphagamma[None, :]
            """ Return the c-centered Voigt line shape at x, scaled to match HWHM of Gaussian and Lorentzian profiles."""

            alpha = 0.61065 * alphagamma
            gamma = 0.61065 * alphagamma
            sigma = alpha / np.sqrt(2 * np.log(2))
            profile = np.real(wofz(((x - c) + 1j * gamma) / (sigma * np.sqrt(2)))) / (sigma * np.sqrt(2 * np.pi))
            profile *= amps
            profile = np.sum(profile, axis=1)
            return profile

        alphagamma = alphagamma * 1 / np.cos(np.arcsin(np.array(q_loc) / np.pi / 2 / 10 * wavelength / 2))
        data = Voigt(q_vectors, q_loc, alphagamma, intensity)  # *intensity.reshape(1,-1)
        weight = SUM_NORM / np.sum(data)
        data = data * weight  # low weight means strong diffraction (Riet-sum coeff * weight = volume fraction)
        return data, weight

    # @staticmethod
    # def get_voigt_xrd(q_vectors, q_loc, intensity, alphagamma, wavelength, v=0, u=0):
    #     def Voigt(x, c, alphagamma, amps):
    #         x = x[:, None]
    #         c = c[None, :]
    #         amps = amps[None, :]
    #         alphagamma = alphagamma[None, :]
    #         """ Return the c-centered Voigt line shape at x, scaled to match HWHM of Gaussian and Lorentzian profiles."""
    #
    #         alpha = 0.61065 * alphagamma
    #         gamma = 0.61065 * alphagamma
    #         sigma = alpha / np.sqrt(2 * np.log(2))
    #         profile = np.real(wofz(((x - c) + 1j * gamma) / (sigma * np.sqrt(2)))) / (sigma * np.sqrt(2 * np.pi))
    #         profile *= amps
    #         profile = np.sum(profile, axis=1)
    #         return profile
    #
    #     theta = np.arcsin(np.array(q_loc) / np.pi / 2 / 10 * wavelength / 2)
    #     w = alphagamma ** 2
    #     alphagamma = np.sqrt(u * np.tan(theta) ** 2 + v * np.tan(theta) + w)
    #     # alphagamma = alphagamma * 1 / np.cos(np.arcsin(np.array(q_loc) / np.pi / 2 / 10 * wavelength / 2))
    #     data = Voigt(q_vectors, q_loc, alphagamma, intensity)  # *intensity.reshape(1,-1)
    #     weight = SUM_NORM / np.sum(data)
    #     data = data * weight  # low weight means strong diffraction (Riet-sum coeff * weight = volume fraction)
    #
    #     return data, weight

    @property
    def entries(self):
        return [_.entry for _ in self.solution]

    def prune_candidates_based_on_composition(self, cutoff=0.05):
        basis_comp = np.array([[e.composition[Element(el)] for el in self.chemsys] for e in self.entries])
        basis_comp /= np.sum(basis_comp, axis=1, keepdims=True)
        diff = basis_comp - self.composition
        ratio = self.composition[np.argmax(diff, axis=1)] / (
                np.max(diff, axis=1) + self.composition[np.argmax(diff, axis=1)])

        self.solution = [self.solution[i] for i in range(len(ratio)) if ratio[i] > cutoff]
        return

    def prune_candidate_based_on_xrd(self, extension=None, plot=False, saveplot=None, cutoff=0.05):
        if not extension:
            extension = self.default_extension

        target = deepcopy(self.exp_xrd)
        target = target / np.max(target) * 100
        target[np.where(target < 1e-6)] = 1e-6

        if plot:
            plt.figure(figsize=(8, 6))
            plt.plot(self.q, target, color='k', linewidth=1)

        # wider_q = np.exp(self.wider_log_q())
        right = np.linspace(self.q[-1], self.q[-1] + (self.q[-1] - self.q[-2]) * extension, extension) + self.q[-1] - \
                self.q[-2]
        left = np.linspace(self.q[0] - extension * (self.q[1] - self.q[0]), self.q[0], extension) - self.q[1] + self.q[
            0]

        wider_q = np.concatenate((left, self.q, right))

        fractions = []
        for phase in self.solution:

            amp, weight = self.get_voigt_xrd(wider_q, *phase.entry.data['xrd'], initial_alphagamma, self.wavelength)
            amp = amp / np.max(amp) * 100

            # plt.plot(wider_q, amp, 'r', alpha=0.6)

            shifted_amps = np.array(
                [np.roll(amp, i)[extension:-extension] for i in range(-extension, extension + 1)])

            shifted_amps = np.maximum(shifted_amps, 1e-6)
            ratio = target.reshape(1, -1) / shifted_amps
            threshold = 10
            mask = np.where(shifted_amps < threshold)
            ratio[mask] = 1

            anchor = np.argmax(np.min(ratio, axis=1))

            # print(target[706], shifted_amps[25][706])
            # for i in range(shifted_amps.shape[0]):
            #    plt.plot(wider_q[extension:-extension], shifted_amps[i])
            # plt.plot(wider_q[extension:-extension], shifted_amps[anchor] * np.max(np.min(ratio, axis=1)))

            # plt.plot(self.q, shifted_amps[anchor] * np.max(np.min(ratio, axis=1)))

            amp = shifted_amps[anchor]

            fraction = np.min(ratio[anchor])  # * np.max(amp) / np.max(self.exp_xrd)

            phase.fraction = fraction
            phase.shift = (anchor - extension) * self.log_q_spacing
            phase.amp = shifted_amps[anchor]
            fractions.append(fraction)



        self.solution = [_ for _ in self.solution if _.fraction > cutoff]
        # seq = sorted(range(len(fractions)), key=lambda k: fractions[k], reverse=True)
        self.solution.sort(key=lambda x: x.fraction)


        if plot:
            for fraction, phase in zip(fractions, self.solution):
                plt.plot(self.q, phase.amp * phase.fraction,
                         # plt.plot(wider_q[extension: -extension], phase.amp * phase.fraction,
                         label='_'.join([phase.entry.name, phase.entry.entry_id, f'{phase.fraction:.3f}']))
            plt.legend(loc=2, bbox_to_anchor=(0, -0.05), ncol=3, prop={'size': 6},
                       title='Sample ' + str(self.sample_id) + '_' + str(self.composition))
            # plt.legend(loc=1, title='Sample ' + str(self.sample_id) + '_' + str(self.composition), ncol=3,
            #            prop={'size': 6})
            if saveplot:
                plt.savefig(saveplot, format='pdf', bbox_inches='tight')
            else:
                plt.show()
            plt.close()
        return

    @property
    def phase_fractions(self):
        # basis_comp = np.array([[Composition(e.name)[el] for el in self.chemsys] for e in self.entries])
        # basis_comp = basis_comp / np.sum(basis_comp, axis=1, keepdims=True)
        basis_xrd_weight = np.array([_.weight for _ in self.solution])
        # print (basis_xrd_weight)
        fractions = np.array([_.fraction for _ in self.solution])
        phase_fractions = basis_xrd_weight * fractions
        phase_fractions /= np.sum(phase_fractions)
        return phase_fractions

    @property
    def recon_comp(self):
        basis_comp = np.array([[Composition(e.name)[el] for el in self.chemsys] for e in self.entries])
        basis_comp = basis_comp / np.sum(basis_comp, axis=1, keepdims=True)
        basis_xrd_weight = np.array([_.weight for _ in self.solution])
        fractions = np.array([_.fraction for _ in self.solution])
        recon_comp = np.sum(basis_comp * fractions.reshape(-1, 1) * basis_xrd_weight.reshape(-1, 1), axis=0)
        recon_comp /= np.sum(recon_comp)
        return recon_comp

    @property
    def comp_loss(self):
        return np.sum((self.composition - self.recon_comp) ** 2)

    @property
    def entropy_loss(self):
        fractions = self.phase_fractions  # np.array([_.fraction for _ in self.solution])
        fractions /= np.sum(fractions)
        return entropy(fractions)

    def loss(self, loss_weight):
        return self.R * loss_weight['xrd_loss'] + \
               self.comp_loss * loss_weight['comp_loss'] + \
               self.entropy_loss * loss_weight['entropy_loss']

    def print_loss(self, loss_weight):
        print(
            f' W-sum | XRD | Comp | Entropy \n {self.loss(loss_weight):.4f}  {self.R:.4f}  {self.comp_loss:.4f}  {self.entropy_loss:.4f}')
        return

    # todo

    @property
    def frac_norm(self):
        return sum([_.fraction for _ in self.solution])

    def to_autoencoder(self, shifts, loss_weight):

        wider_q = np.exp(self.wider_log_q(extension=shifts))
        basis_xrds_np, basis_xrd_weight_np = [], []

        for phase in self.solution:
            qloc, amp = phase.entry.data['xrd']
            qloc = qloc * (1 + phase.shift)
            amp, weight = self.get_voigt_xrd(wider_q, qloc, amp, phase.width, self.wavelength)
            basis_xrds_np.append(amp)
            basis_xrd_weight_np.append(weight)

        basis_xrds_np = np.array(basis_xrds_np)
        sample_xrd_np = self.exp_xrd.reshape(1, -1) / self.frac_norm
        basis_xrds_np = basis_xrds_np
        sample_xrd_np = sample_xrd_np
        basis_xrd_weight_np = np.array(basis_xrd_weight_np)

        basis_comp_np = np.array([[Composition(e.name)[el] for el in self.chemsys] for e in self.entries])
        basis_comp_np = basis_comp_np / np.sum(basis_comp_np, axis=1, keepdims=True)
        sample_comp_np = self.composition

        basis_xrd = th.from_numpy(basis_xrds_np).float().cpu()
        sample_xrd = th.from_numpy(sample_xrd_np).float().cpu()

        basis_comp = th.from_numpy(basis_comp_np).float().cpu()
        basis_xrd_weight = th.from_numpy(basis_xrd_weight_np).float().cpu()
        sample_comp = th.from_numpy(sample_comp_np).float().cpu()

        model = Autoencoder(basis_xrd, sample_xrd, basis_xrd_weight, basis_comp, sample_comp, shifts, loss_weight)
        return model

    def optimize(self, loss_weight, num_epoch=1000, shifts=15, print_prog=False):
        new_model = deepcopy(self)

        model = new_model.to_autoencoder(shifts, loss_weight)
        model.train(num_epoch, print_prog=print_prog)
        fractions, shifts, stds = model.solution()

        for i in range(len(new_model.solution)):
            new_model.solution[i].shift += shifts[i] * new_model.log_q_spacing
            new_model.solution[i].fraction = fractions[i] * self.frac_norm
            new_model.solution[i].width += stds[i] * (new_model.q[-1] - new_model.q[0]) / len(new_model.q)

        new_model.solution = sorted(new_model.solution, key=lambda x: x.fraction, reverse=True)

        if new_model.loss(loss_weight) < self.loss(loss_weight):
            return new_model
        else:
            return self

    def update_solution(self, frac_cutoff, width_cutoff, shift):
        self.solution = [_ for _ in self.solution if _.fraction > frac_cutoff]
        self.solution = [_ for _ in self.solution if _.width < width_cutoff - 1e-8]
        self.solution = [_ for _ in self.solution if abs(_.shift) < shift - 1e-8]

    def print_solution(self):
        self.solution.sort(key=lambda x: x.fraction, reverse=True)
        data = {
            'Name': [_.entry.name for _ in self.solution],
            'Entry_id': [_.entry.entry_id for _ in self.solution],
            'fraction': self.phase_fractions,
            'shift': [_.shift for _ in self.solution],
            'width': [_.width for _ in self.solution],
        }
        df = pd.DataFrame(data)
        # df.sort_values(by=['fraction'], ascending=False, inplace=True)
        # df.reset_index(drop=True, inplace=True)
        print(f'Sample: # {self.sample_id}')
        print(df)
        print(f'Current R^2 = {self.R}')

        return

    @property
    def current_model(self):
        ys = []
        for phase in self.solution:
            e = phase.entry
            q_locs, amps = e.data['xrd']
            shift = phase.shift
            width = phase.width
            fraction = phase.fraction
            y, _ = self.get_voigt_xrd(self.q, q_locs * (1 + shift), amps, width, self.wavelength)
            y = y * fraction
            ys.append(y)
        ys = np.array(ys)
        return ys

    def refine_param(self, index):
        # index is the index of phase in solution
        x_data = self.q

        ys = self.current_model
        y_data = self.exp_xrd - np.sum(ys, axis=0)
        y_data += ys[index]

        fraction = self.solution[index].fraction
        width = self.solution[index].width
        shift = self.solution[index].shift

        q_locs, amps = self.solution[index].entry.data['xrd']

        def fit_function(sw):
            shift, width = sw
            recon, _ = self.get_voigt_xrd(x_data, q_locs * (1 + shift), amps, width, self.wavelength)
            recon = fraction * recon
            recon += np.sum(ys, axis=0)
            recon -= ys[index]
            w = 1 / (self.exp_xrd + 0.01 * np.max(self.exp_xrd))
            R = np.sum(w * (recon - self.exp_xrd) ** 2) / np.sum(self.exp_xrd)
            return R

        # print(self.max_q_shift)
        # print(shift, width)
        # res = minimize(fit_function, x0=[shift, width], bounds=[(-self.max_q_shift, self.max_q_shift), (0, 0.3)])

        res = minimize(fit_function, x0=np.array([shift, width]),
                       bounds=([-self.max_q_shift, self.max_q_shift], [0, 0.3]))

        self.solution[index].shift, self.solution[index].width = res.x

    def refine_one_by_one(self):
        for index in range(len(self.solution)):
            self.refine_param(index)

    def refine_all_fractions(self):
        # I got to do a wrapper here, because I do not know the number of fitting parameters...
        # Solution comes from this post: https://stackoverflow.com/questions/38327846/using-undetermined-number-of-parameters-in-scipy-function-curve-fit

        fractions = np.array([_.fraction for _ in self.solution])
        ys = self.current_model
        unscaled_ys = ys / fractions.reshape(-1, 1)

        def fit_function(fracs):
            recon = np.zeros(len(self.q))
            for frac, y in zip(fracs, unscaled_ys):
                recon += frac * y
            w = 1 / (self.exp_xrd + 0.01 * np.max(self.exp_xrd))
            R = np.sum(w * (recon - self.exp_xrd) ** 2) / np.sum(self.exp_xrd)
            return R

        res = minimize(fit_function, x0=fractions, bounds=[(0, np.inf) for _ in fractions])  # method='bound'
        for i, frac in enumerate(res.x):
            self.solution[i].fraction = frac

    @property
    def recon(self):
        recon = np.sum(self.current_model, axis=0)
        return recon

    @property
    def R(self):
        # Figured out this is R squred actually ......

        w = 1 / (self.exp_xrd + 0.01 * np.max(self.exp_xrd))
        # w /= np.sum(w)
        # w *= len(w)
        # print (w)

        R = np.sum(w * (self.recon - self.exp_xrd) ** 2) / np.sum(self.exp_xrd)
        return R

    def plot(self, show=False, saveplot=None, perphase=False):
        if perphase:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8 * 2, 6), sharey=True)
            for phase, y in zip(self.solution, self.current_model):
                if phase.fraction:
                    ax2.plot(self.q, y, label=phase.entry.entry_id + ' ' + phase.entry.name, lw=1)
            ax2.legend()
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))

        ax1.plot(self.q, self.exp_xrd, label='orig.', lw=1)
        ax1.plot(self.q, self.recon, label='recon.', alpha=0.5, lw=1)
        ax1.legend(title=f'#{self.sample_id:03d} R={np.sqrt(self.R):.3f}')

        if saveplot:
            plt.savefig(saveplot, format='pdf', bbox_inches='tight', transparent=True)
            plt.close()
        elif show:
            plt.show()
        else:
            return plt


class Phase(MSONable):
    def __init__(self, entry, fraction, shift, width, q, amp, weight):
        self.entry = entry
        self.fraction = fraction
        self.shift = shift
        self.width = width
        self.q = q
        self.amp = amp
        self.weight = weight

    @classmethod
    def from_entry_and_instance_data(cls, entry, fraction, instance_data, shift=0, width=initial_alphagamma):
        amp, weight = Sample.get_voigt_xrd(instance_data.q, *entry.data['xrd'], width, instance_data.wavelength)
        return cls(entry, fraction, shift, width, instance_data.q, amp, weight)


def main():
    instance_data = InstanceData.from_file('../data/instance_file_24297_NbMnVO_v02.txt', chemsys, photon_e)
    instance_data = instance_data.resample_xrd(resample_density)
    instance_data.renormalize(norm=SUM_NORM)

    with open('../data/icdd_entries.json') as f:
        entries = json.load(f, cls=MontyDecoder)

    solved = [int(_[16:-5]) for _ in glob("solution/samples*.json")]
    solved.sort()
    unsolved = [_ for _ in range(instance_data.sample_num) if _ not in solved]
    unsolved.sort()

    for i in range(instance_data.sample_num):
        solution_file = f'solution/samples{i}.json'
        if os.path.exists(solution_file):
            with open(solution_file) as f:
                sample = json.load(f, cls=MontyDecoder)
            print(i, sample.R)
        else:
            print(f'Solving sample {i} ......')
            solution = []
            for e in entries:
                phase = Phase.from_entry_and_instance_data(e, 1 / len(entries), instance_data)
                solution.append(phase)

            sample = Sample(i, instance_data.log_q, instance_data.sample_xrd[i], instance_data.chemsys,
                            instance_data.sample_comp[i], oxide_system, instance_data.wavelength, max_q_shift, solution)

            sample.prune_candidates_based_on_composition(cutoff=0.05)
            sample.prune_candidate_based_on_xrd(plot=True, saveplot=f'initial_pruning/fig_{i}.pdf')

            sample = sample.optimize(num_epoch=500, print_prog=False)
            sample.update_solution(0.03, 0.2999)

            sample = sample.optimize(num_epoch=1000, print_prog=False)
            sample.update_solution(0.03, 0.2999)
            sample.refine_one_by_one()

            sample.print_solution()
            sample.print_loss()

            with open(solution_file, 'w') as f:
                json.dump(sample, f, cls=MontyEncoder)


if __name__ == "__main__":
    main()
