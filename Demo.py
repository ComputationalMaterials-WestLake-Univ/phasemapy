# Automated phase mapping of high throughput X-ray diffraction data
'''
This is an example of automated phase mapping on high throughput X-ray diffraction data.The process is divided into 4 steps as follows:

1.Load the instance data and generate ICDD entries pool
2.Discard the unreasonable candidates based on DFT calculation
3.Prune candidates based on composition and XRD
4.XRD fitting and solution optimization based on Autoencoder

'''

import json, os
import numpy as np
from monty.json import MontyDecoder, MontyEncoder
from copy import deepcopy

from phasemapy.dataio import InstanceData
from phasemapy.parser import ICDDEntry
from phasemapy.solver import Phase, Sample

chemsys = ['Bi', 'Cu', 'V']
oxide_system = True
photon_e = 13e3
max_q_shift = 0.05
resample_density = 1000
initial_alphagamma = 0.1
SUM_NORM = 6000
loss_weight = {'xrd_loss': 6.0, 'comp_loss': 2.0, 'entropy_loss': 0.01}

# 1.Load the instance data and ICDD entries pool in Bi-Cu-V system

instance_data = InstanceData.from_json('C:/Users/dell/Desktop/phasemapy/scripts_Bi_Cu_V_O/data/Instance_data',
                                             chemsys, photon_e) #load instance data:307 instances

with open('C:/Users/dell/Desktop/phasemapy/scripts_Bi_Cu_V_O/data/ICDD_entries_merge_by_xrd.json') as f:
    entries = json.load(f, cls=MontyDecoder)
entries = [ICDDEntry.from_icdd_json(en) for en in entries] #load ICDD entries pool: 78 ICDD entries

# 2.Discard the wrong candidates based on DFT calculation
# 3.Prune candidates based on composition and XRD. and then choose the first instance as example.
i = 0
solution = []
for e in entries:
    phase = Phase.from_entry_and_instance_data(e, 1 / len(entries), instance_data)
    solution.append(phase)

sample = Sample(i, instance_data.log_q, instance_data.sample_xrd[i], instance_data.chemsys,
                instance_data.sample_comp[i], oxide_system, instance_data.wavelength, max_q_shift, solution)
sample.prune_candidates_based_on_composition(cutoff=0.05)
sample.prune_candidate_based_on_xrd(plot=False, cutoff=0.005)

sample.refine_all_fractions()
sample.update_solution_new(0.03, 0.2999, sample.max_q_shift)
sample.refine_one_by_one()
sample.plot(perphase=True)

# 4.XRD fitting and solution optimization based on Autoencoder
new_sample = deepcopy(sample)
new_sample = new_sample.optimize(num_epoch=500, print_prog=False, loss_weight=loss_weight)
new_sample.update_solution_new(0.12, 0.2999, new_sample.max_q_shift)
new_sample.refine_all_fractions()
new_sample.refine_one_by_one()
new_sample.print_solution_new()
sample.plot(perphase=True)