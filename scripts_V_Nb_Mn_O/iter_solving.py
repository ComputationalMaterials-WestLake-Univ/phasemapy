import os
import json
import numpy as np
from copy import deepcopy

from collections import defaultdict
from monty.json import MontyDecoder, MSONable, MontyEncoder
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
import sys

from phasemapy.dataio import InstanceData
from phasemapy.parser import ICDDEntry
from phasemapy.solver import Phase, Sample

sys.path += ['/Users/yizhou/PycharmProjects/phasemapy/phasemapy']

chemsys = ['V', 'Mn', 'Nb']
oxide_system = True
photon_e = 13e3
max_q_shift = 0.02
resample_density = 2000
initial_alphagamma = 0.03
SUM_NORM = 6000
loss_weight = {'xrd_loss': 6.0, 'comp_loss': 2.0, 'entropy_loss': 0.2}

instance_data = InstanceData.from_file('../data/instance_file_24297_NbMnVO_v02.txt', chemsys, photon_e)
# instance_data = instance_data.resample_xrd(resample_density)
instance_data.renormalize(norm=SUM_NORM)

comp_dist = distance_matrix(instance_data.sample_comp, instance_data.sample_comp)
nn_list = {i: np.where((comp_dist[i] < 0.15) & (comp_dist[i] > 0))[0] for i in range(instance_data.sample_num)}
samples = []

for i in range(instance_data.sample_num):
    solution_file = f'solution/samples{i}.json'
    if os.path.exists(solution_file):
        with open(solution_file) as f:
            sample = json.load(f, cls=MontyDecoder)
            samples.append(sample)

# os.mkdir('solution_figures')
for sample in samples:
    sample.plot(perphase=True, saveplot=f'solution_figures/sample_{sample.sample_id}.pdf')

exit()
# j = 0
# solution_file = f'solution/samples{j}.json'
# if os.path.exists(solution_file):
#     with open(solution_file) as f:
#         orig_sample = json.load(f, cls=MontyDecoder)
# orig_sample.plot(perphase=True)


for sample in samples:
    if True:
        # if min([count_act[_.entry_id] for _ in sample.entries])<5.0:
        # if sample.R > 0.4:
        print(sample.sample_id, sample.loss(loss_weight))
        # candidate_entries = []
        # for i in nn_list[sample.sample_id]:
        #     candidate_entries += samples[i].entries
        # candidate_entries = list(set(candidate_entries))
        # solution = []
        # for e in candidate_entries:
        #     phase = Phase.from_entry_and_instance_data(e, 1 / len(candidate_entries), instance_data)
        #     solution.append(phase)

        new_sample = deepcopy(sample)
        # new_sample.solution = solution
        # au = new_sample.to_auto
        new_sample = new_sample.optimize(num_epoch=500, print_prog=True, loss_weight=loss_weight)
        new_sample.update_solution(0.03, 0.2999, new_sample.max_q_shift)

        new_sample = new_sample.optimize(num_epoch=3000, print_prog=True, loss_weight=loss_weight)
        new_sample.update_solution(0.01, 0.2999, new_sample.max_q_shift)

        new_sample.refine_one_by_one()
        new_sample.refine_all_fractions()

        new_sample.update_solution(0.01, 0.2999, new_sample.max_q_shift)
        print(new_sample.loss(loss_weight))
        if new_sample.loss(loss_weight) <= sample.loss(loss_weight):
            sample.print_solution()
            new_sample.print_solution()
            samples[sample.sample_id] = new_sample
            solution_file = f'solution/samples{sample.sample_id}.json'
            with open(solution_file, 'w') as f:
                json.dump(sample, f, cls=MontyEncoder)
