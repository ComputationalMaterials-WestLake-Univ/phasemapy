import json
import pandas as pd
from copy import deepcopy
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
import os
from monty.json import MontyEncoder
from pymatgen.core import Element

from phasemapy.parser import ICDDEntry, ICDDEntryPreprocessor

chemsys = ['Bi', 'Cu', 'V']
oxide_system = True


def get_dataframe(icdd_entries, keys):
    data = {}
    for key in keys:
        data[key] = [e.as_dict()[key] for e in icdd_entries]
    df = pd.DataFrame(data)
    return df

# Load the entries of ICDD
pdfs_icdd = glob('./data/icdd/*.xml')
icdd_entries = [ICDDEntry.from_icdd_xml(pdf) for pdf in pdfs_icdd]
icdd_entries = [_ for _ in icdd_entries if _.name != 'O2']
icdd_entries = [_ for _ in icdd_entries if _.name != 'O3']
icdd_entries = [_ for _ in icdd_entries if _.name != '(O3)']
icdd_entries = [_ for _ in icdd_entries if _.name != '(O2)']
precess = ICDDEntryPreprocessor(deepcopy(icdd_entries), chemsys, oxide_system)

# Load the entries of ICSD
pdfs_icsd = glob('./data/icsd/*.cif')
icsd_entries = [ICDDEntry.from_icsd_cif(pdf) for pdf in pdfs_icsd]
precess_icsd = ICDDEntryPreprocessor(deepcopy(icsd_entries), chemsys, oxide_system)
precess_icsd.get_xrd()
# precess_icsd.process_frac_name()
# precess_icsd.process_disorder()
icsd_entries = precess_icsd.entries

#Remove the Deleted, Hypothetical, Blank, Low-Precision, non-Ambient entries of ICDD
print('[ICDD] Total (Bi-Cu-V) - O: ', len(icdd_entries))  # Total

icdd_entries = [_ for _ in icdd_entries if _.status != 'Deleted']
print('[ICDD] after remove Deleted:', len(icdd_entries))

icdd_entries = [_ for _ in icdd_entries if _.quality_mark != 'Hypothetical']
print('[ICDD] after remove Hypothetical:', len(icdd_entries))

icdd_entries = [_ for _ in icdd_entries if _.quality_mark not in ['Blank', 'Low-Precision']]
print('[ICDD] after remove Blank/Low-Precision:', len(icdd_entries))

icdd_entries = [_ for _ in icdd_entries if _.pressure_temperature == 'Ambient']
print('[ICDD] after remove non-Ambient:', len(icdd_entries))



#Merging cross-ref entries of ICDD
precess = ICDDEntryPreprocessor(deepcopy(icdd_entries), chemsys, oxide_system)
precess.merge_by_cross_ref()
icdd_entries= precess.entries
print('[ICDD] after merging cross-ref entries', len(icdd_entries))

#Merging duplicate entries of ICSD and ICDD
icdd_entries = precess.entries
precess = ICDDEntryPreprocessor(deepcopy(icdd_entries), chemsys, oxide_system)
precess.theta_to_q()
precess.get_xrd()
icdd_entries = precess.entries
entries = icdd_entries + icsd_entries
precess = ICDDEntryPreprocessor(deepcopy(entries), chemsys, oxide_system)
groups=precess.merge_by_xrd(bin_number=1000, gaussian_filter=4, R1_cutoff=0.2, R2_cutoff=0.35, plot=False)
print('[ICDD] after merging XRD-group entries', len(precess.entries))
print('[Groups] after merging XRD-group entries', groups[0])
print('[Groups] after merging XRD-group entries', groups[1])

#Remove unstable entries
null_stability_entries, stable_entries,unstable_entries, candidates_entires = precess.check_stability(0.1)

df_null_stability_entries = get_dataframe([_ for _ in null_stability_entries],
                   ['entry_id', 'name', 'pressure_temperature', 'cross_refs', 'status', 'quality_mark', 'name',
                    'spgr', 'common_name', 'leader',  'stability'])

df_stable_entries = get_dataframe([_ for _ in stable_entries],
                   ['entry_id', 'name', 'pressure_temperature', 'cross_refs', 'status', 'quality_mark', 'name',
                    'spgr', 'common_name', 'leader',  'stability'])
df_unstable_entries = get_dataframe([_ for _ in unstable_entries],
                   ['entry_id', 'name', 'pressure_temperature', 'cross_refs', 'status', 'quality_mark', 'name',
                    'spgr', 'common_name', 'leader',  'stability'])
df_candidates = get_dataframe([_ for _ in candidates_entires],
                   ['entry_id', 'name', 'pressure_temperature', 'cross_refs', 'status', 'quality_mark', 'name',
                    'spgr', 'common_name', 'leader',  'stability'])
print(df_candidates)
print('[ICDD] entries without stability', [_.entry_id for _ in null_stability_entries])
print('[ICDD] entries with stability', df_candidates)
df_null_stability_entries.to_excel('./data/df_null_stability_entries.xlsx')
df_stable_entries.to_excel('./data/output_stable_pool.xlsx')
df_unstable_entries.to_excel('./data/output_unstable_pool.xlsx')
df_candidates.to_excel('./data/output_candidates_pool.xlsx')

with open('./data/entries_dft_sno2.json', 'w') as f:
        json.dump(candidates_entires, f, cls=MontyEncoder)


