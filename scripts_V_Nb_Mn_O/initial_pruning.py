import json, os
import numpy as np
from monty.json import MontyDecoder, MontyEncoder
from pymatgen.analysis.diffraction.xrd import XRDCalculator

from phasemapy.dataio import InstanceData
from phasemapy.parser import ICDDEntry
from phasemapy.solver import Phase, Sample

chemsys = ['V', 'Mn', 'Nb']
oxide_system = True
photon_e = 13e3
max_q_shift = 0.02
resample_density = 2000
initial_alphagamma = 0.03
SUM_NORM = 6000
loss_weight = {'xrd_loss': 6.0, 'comp_loss': 2.0, 'entropy_loss': 0.2}
instance_data = InstanceData.from_file('../data/instance_file_24297_NbMnVO_v02.txt', chemsys, photon_e)
instance_data = instance_data.resample_xrd(resample_density)
instance_data.renormalize(norm=SUM_NORM)

with open('../data/icdd_oqmd_entries.json') as f:
    entries = json.load(f, cls=MontyDecoder)
print(len(entries))

for e in entries:
    xrdcal = XRDCalculator()
    s = e.structure
    xrd = xrdcal.get_pattern(s, scaled=False)
    d = np.array(xrd.d_hkls)
    amplitude = np.array(xrd.y) / s.volume ** 2  # This is irrelevant of volume
    q_vectors = 4 * np.pi / (2 * d) * 10
    data = (q_vectors, amplitude)
    e.data['xrd'] = data


# initial pruning
# os.mkdir('initial_pruning')
# os.mkdir('solution')

for i in range(instance_data.sample_num):
    solution_file = f'solution/samples{i}.json'
    if os.path.exists(solution_file):
        continue
        with open(solution_file) as f:
            sample = json.load(f, cls=MontyDecoder)
        # print(i, sample.R)
        continue
    else:
        print(f'Solving sample {i} ......')
        solution = []
        for e in entries:
            phase = Phase.from_entry_and_instance_data(e, 1 / len(entries), instance_data)
            solution.append(phase)

        sample = Sample(i, instance_data.log_q, instance_data.sample_xrd[i], instance_data.chemsys,
                        instance_data.sample_comp[i], oxide_system, instance_data.wavelength, max_q_shift, solution)
        # sample.print_solution()

        sample.prune_candidates_based_on_composition(cutoff=0.05)
        sample.prune_candidate_based_on_xrd(plot=True, cutoff=0.05, saveplot=f'initial_pruning/fig_{i}.pdf')
        # sample.print_solution()
        #             sample = sample.optimize(num_epoch=500, print_prog=False)
        #             sample.update_solution(0.03, 0.2999)

        #             sample = sample.optimize(num_epoch=1000, print_prog=False)
        #             sample.update_solution(0.03, 0.2999)
        #             sample.refine_one_by_one()

        #             sample.print_solution()
        #             sample.print_loss()

        with open(solution_file, 'w') as f:
            json.dump(sample, f, cls=MontyEncoder)
