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

chemsys = ['Li', 'Sr', 'Al']
oxide_system = True

def main():
    def get_dataframe(icdd_entries, keys):
        data = {}
        for key in keys:
            data[key] = [e.as_dict()[key] for e in icdd_entries]
        df = pd.DataFrame(data)
        return df

    pdfs_icdd = glob('./data/icdd/*.xml')
    icdd_entries = [ICDDEntry.from_icdd_xml(pdf) for pdf in pdfs_icdd]
    icdd_entries = [_ for _ in icdd_entries if _.name != 'O2']
    precess = ICDDEntryPreprocessor(deepcopy(icdd_entries), chemsys, oxide_system)

    pdfs_icsd = glob('./data/icsd/*.cif')
    icsd_entries = [ICDDEntry.from_icsd_cif(pdf) for pdf in pdfs_icsd]
    precess_icsd = ICDDEntryPreprocessor(deepcopy(icsd_entries), chemsys, oxide_system)
    precess_icsd.get_xrd()
    icsd_entries = precess_icsd.entries


    df = get_dataframe(precess.entries,
                       ['entry_id', 'name', 'pressure_temperature', 'cross_refs', 'status', 'quality_mark', 'name',
                        'spgr', 'common_name'])

    print('[ICDD] Total (Li-Sr-Al) - O: ', len(icdd_entries))  # Total

    icdd_entries = [_ for _ in icdd_entries if _.status != 'Deleted']
    print('[ICDD] after remove Deleted:', len(icdd_entries))

    icdd_entries = [_ for _ in icdd_entries if _.quality_mark != 'Hypothetical']
    print('[ICDD] after remove Hypothetical:', len(icdd_entries))

    icdd_entries = [_ for _ in icdd_entries if _.quality_mark not in ['Blank', 'Low-Precision']]
    print('[ICDD] after remove Blank/Low-Precision:', len(icdd_entries))

    icdd_entries = [_ for _ in icdd_entries if _.pressure_temperature == 'Ambient']
    print('[ICDD] after remove non-Ambient:', len(icdd_entries))

#     icdd_entries = [_ for _ in icdd_entries if _.structure]
#     print('[ICDD] after remove no-struct entries', len(icdd_entries))

    # icdd_entries = [_ for _ in icdd_entries if check_oxi(_.composition,chemsys)]
    # print('[ICDD] after remove weird-valence entries', len(icdd_entries))

    precess = ICDDEntryPreprocessor(deepcopy(icdd_entries), chemsys, oxide_system)
    precess.process_frac_name()
    precess.process_disorder()
    precess.merge_by_cross_ref()
    print('[ICDD] after merging cross-ref entries', len(precess.entries))
    precess.get_xrd()
    precess.merge_by_polymorph(bin_number=1000, gaussian_filter=4, R_cutoff=0.2)
    print('[ICDD] after merging XRD-polymorph entries', len(precess.entries))
    precess.merge_by_xrd(bin_number=1000, gaussian_filter=4, R_cutoff=0.22)
    print('[ICDD] after merging XRD-group entries', len(precess.entries))

    print(len([_ for _ in precess.entries if _.structure.is_ordered]), 'ordered structures')
    print(len([_ for _ in precess.entries if not _.structure.is_ordered]), 'disordered structures')

    all_entries = precess.entries
    entries_nostruct = [_ for _ in all_entries  if _.structure == None]
    # if len(entries_nostruct) != 0:
    #     # all_entries = precess.check_structure(R_cutoff)
    #     pass
    nostability_entries, all_entries = precess.check_stability(0.3)
    df = get_dataframe([_ for _ in all_entries ],
                       ['entry_id', 'name', 'pressure_temperature', 'cross_refs', 'status', 'quality_mark', 'name',
                        'spgr', 'common_name', 'leader', 'stability'])
    print(df)
    print('[ICDD] entries without stability', [_.entry_id for _ in nostability_entries])
    df.to_excel('./data/output_candidate_pool.xlsx')

    with open('./data/entries_dft.json', 'w') as f:
            json.dump(all_entries, f, cls=MontyEncoder)



if __name__ == "__main__":
    main()

