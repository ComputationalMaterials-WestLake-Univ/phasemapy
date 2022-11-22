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

    def check_oxi(comp, chemsys):
        c1 = comp[Element(chemsys[0])] * np.min(Element(chemsys[0]).common_oxidation_states) \
             + comp[Element(chemsys[1])] * np.min(Element(chemsys[1]).common_oxidation_states) \
             + comp[Element(chemsys[2])] * np.min(Element(chemsys[2]).common_oxidation_states) \
             - comp[Element('O')] * 2
        c2 = comp[Element(chemsys[0])] * np.max(Element(chemsys[0]).common_oxidation_states) \
             + comp[Element(chemsys[1])] * np.max(Element(chemsys[1]).common_oxidation_states) \
             + comp[Element(chemsys[2])] * np.max(Element(chemsys[2]).common_oxidation_states) \
             - comp[Element('O')] * 2
        return c1 * c2 <= 0

    pdfs = glob('./data/icdd/*.xml')
    icdd_entries = [ICDDEntry.from_icdd_xml(pdf) for pdf in pdfs]
    icdd_entries = [_ for _ in icdd_entries if _.name != 'O2']

    precess = ICDDEntryPreprocessor(deepcopy(icdd_entries), chemsys, oxide_system)
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

    icdd_entries = [_ for _ in icdd_entries if check_oxi(_.composition,chemsys)]
    print('[ICDD] after remove weird-valence entries', len(icdd_entries))

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
#     print(len([_ for _ in precess.entries if _.structure.composition.as_dict().keys() == {'V', 'O'}]))



    all_entries = precess.entries
    entries_nostruct = [_ for _ in all_entries  if _.structure == None]
    if len(entries_nostruct) == 0:
        df = get_dataframe([_ for _ in all_entries ],
                       ['entry_id', 'name', 'pressure_temperature', 'cross_refs', 'status', 'quality_mark', 'name',
                        'spgr', 'common_name', 'leader'])
        print(df)
        df.to_excel('./data/output_candidate_pool.xlsx')

        with open('./data/icdd_entries.json', 'w') as f:
            json.dump(all_entries, f, cls=MontyEncoder)

    else:
        pass

if __name__ == "__main__":
    main()

