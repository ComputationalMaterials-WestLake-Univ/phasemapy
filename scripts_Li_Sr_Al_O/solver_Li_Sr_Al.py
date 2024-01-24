import warnings
warnings.filterwarnings('ignore')
import json, os
import numpy as np
from monty.json import MontyDecoder, MontyEncoder
from copy import deepcopy
from scipy.spatial import distance_matrix
import sys
sys.path.append('..')

from phasemapy.dataio import InstanceData
from phasemapy.parser import ICDDEntry
from phasemapy.solver import Phase, Sample

chemsys = ['Li', 'Sr', 'Al']
oxide_system = True
photon_e = 13e3
max_q_shift = 0.01
resample_density = 1000
initial_alphagamma = 0.1
SUM_NORM = 6000
loss_weight = {'xrd_loss': 6.0, 'comp_loss': 2.0, 'entropy_loss': 0.01}

#load instance data:50 instances
instance_data = InstanceData.from_json('./data/Instance_data', chemsys, photon_e)
instance_data = instance_data.resample_xrd(resample_density)
instance_data.renormalize(norm=SUM_NORM)
instance_data.normalize()
#load candidates pool:19  entries
with open('./data/entries_dft.json') as f:
    entries = json.load(f, cls=MontyDecoder)
Phase.theta_to_q(entries)

# Solved by AutoEncoder
samples = []
for i in range(instance_data.sample_num):
    solution = []
    for e in entries:
        phase = Phase.from_entry_and_instance_data(e, 1 / len(entries), instance_data)
        solution.append(phase)

    sample = Sample(i, instance_data.log_q, instance_data.sample_xrd[i], instance_data.chemsys,
                    instance_data.sample_comp[i], oxide_system, instance_data.wavelength, max_q_shift, solution)
    sample.prune_candidates_based_on_composition(cutoff=0.005)
    sample.prune_candidate_based_on_xrd(plot=False, cutoff=0.015)
    sample.refine_all_fractions()
    sample.update_solution(0.03, 0.2999, sample.max_q_shift)
    sample = sample.optimize(num_epoch=500, print_prog=True, loss_weight=loss_weight)
    sample.update_solution(0.1, 0.2999, sample.max_q_shift)
    sample.plot(perphase=True)
    samples.append(sample)

for sample in samples:
    sample.update_solution(0.12,0.2999, sample.max_q_shift)

comp_dist = distance_matrix(instance_data.sample_comp, instance_data.sample_comp)
nn_list = {i: np.where((comp_dist[i] < 1) & (comp_dist[i] > 0))[0] for i in range(instance_data.sample_num)}
new_samples=deepcopy(samples)
samples_neighbor = []
for sample in new_samples:
    if len(sample.phase_fractions)==0:
        sample.solution = new_samples[sample.sample_id-1].solution
    if len(sample.phase_fractions)>0:
        candidate_entries = []
        for i in nn_list[sample.sample_id]:
            candidate_entries += new_samples[i].entries
        candidate_entries = list(set(candidate_entries))
        solution = []
        for e in candidate_entries:
            phase = Phase.from_entry_and_instance_data(e, 1 / len(candidate_entries), instance_data)
            solution.append(phase)
    new_sample = deepcopy(sample)
    new_sample.solution = solution
    new_sample.refine_all_fractions()
#     new_sample.refine_one_by_one()
    new_sample.print_solution()
    new_sample = new_sample.optimize(num_epoch=500, print_prog=False,loss_weight=loss_weight)
    new_sample.update_solution(0.001, 0.2999, new_sample.max_q_shift)
    new_sample.refine_one_by_one()
    new_sample.refine_all_fractions()
    new_sample.print_solution()
    new_sample.plot(perphase=True)
    samples_neighbor.append(new_sample)

for sample in samples_neighbor:
    solution_file = f'solution/without_TextureAnalysis/samples{sample.sample_id}.json'
    with open(solution_file, 'w') as f:
        json.dump(samples_neighbor[sample.sample_id], f, cls=MontyEncoder)

#Refinement by Texture Analysis
samples = []
for i in range(instance_data.sample_num):
    solution_file = f'solution/without_TextureAnalysis/samples{i}.json'
    with open(solution_file) as f:
        sample = json.load(f, cls=MontyDecoder)
    if sample.sample_id !=i:
        print (i)
    samples.append(sample)