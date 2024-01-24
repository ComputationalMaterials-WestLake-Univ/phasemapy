
#sys.path.append('/Users/yizhou/PycharmProjects/phasemapy')
import json
import pandas as pd
from copy import deepcopy
from glob import glob

from monty.json import MontyEncoder
from pymatgen.core import Element

from phasemapy.parser import ICDDEntry, ICDDEntryPreprocessor


chemsys = ['V', 'Mn', 'Nb']
oxide_system = True

def main():
    def get_dataframe(icdd_entries, keys):
        data = {}
        for key in keys:
            data[key] = [e.as_dict()[key] for e in icdd_entries]
        df = pd.DataFrame(data)
        return df

    pdfs = glob('../data/icdd/**/*.xml')
    icdd_entries = [ICDDEntry.from_icdd_xml(pdf) for pdf in pdfs]
    icdd_entries = [_ for _ in icdd_entries if _.name != 'O2']

    precess = ICDDEntryPreprocessor(deepcopy(icdd_entries), chemsys, oxide_system)
    df = get_dataframe(precess.entries,
                       ['entry_id', 'name', 'pressure_temperature', 'cross_refs', 'status', 'quality_mark', 'name',
                        'spgr', 'common_name'])
    # print (df[df['name']=='MnV2O6'])

    # df.to_excel("raw_icdd.xlsx")
    # exit()

    print('[ICDD] Total (V-Nb-Mn) - O: ', len(icdd_entries))  # Total

    icdd_entries = [_ for _ in icdd_entries if _.status != 'Deleted']
    print('[ICDD] after remove Deleted:', len(icdd_entries))

    icdd_entries = [_ for _ in icdd_entries if _.quality_mark != 'Hypothetical']
    print('[ICDD] after remove Hypothetical:', len(icdd_entries))

    icdd_entries = [_ for _ in icdd_entries if _.quality_mark not in ['Blank', 'Low-Precision']]
    print('[ICDD] after remove Blank/Low-Precision:', len(icdd_entries))

    icdd_entries = [_ for _ in icdd_entries if _.pressure_temperature == 'Ambient']
    print('[ICDD] after remove non-Ambient:', len(icdd_entries))

    icdd_entries = [_ for _ in icdd_entries if _.structure]
    print('[ICDD] after remove no-struct entries', len(icdd_entries))

    def check_oxi(comp):
        # comp = {el.symbol: comp[el] for el in comp}
        c1 = comp[Element('V')] * 2 + comp[Element('Mn')] * 2 + comp[Element('Nb')] * 2 - comp[Element('O')] * 2
        c2 = comp[Element('V')] * 5 + comp[Element('Mn')] * 4 + comp[Element('Nb')] * 5 - comp[Element('O')] * 2

        return c1 * c2 <= 0

    icdd_entries = [_ for _ in icdd_entries if check_oxi(_.composition)]
    print('[ICDD] after remove weird-valence entries', len(icdd_entries))

    precess = ICDDEntryPreprocessor(deepcopy(icdd_entries), chemsys, oxide_system)
    precess.process_frac_name()
    precess.process_disorder()

    # df = get_dataframe([_ for _ in precess.entries if _.name == 'Mn2V2O7'],
    #                    ['entry_id', 'name', 'pressure_temperature', 'cross_refs', 'status', 'quality_mark', 'name',
    #                     'spgr', 'common_name', 'leader'])
    # print(df)
    sp = [_ for _ in precess.entries if _.entry_id == '00-052-1266'][0]  # This is right Mn2V2O7 beta phase


    precess.merge_by_cross_ref()
    print('[ICDD] after merging cross-ref entries', len(precess.entries))
    # df = get_dataframe([_ for _ in precess.entries if _.name == 'Mn2V2O7'],
    #                    ['entry_id', 'name', 'pressure_temperature', 'cross_refs', 'status', 'quality_mark', 'name',
    #                     'spgr', 'common_name', 'leader'])
    # print(df)


    precess.get_xrd()
    precess.merge_by_polymorph(bin_number=1000, gaussian_filter=4, R_cutoff=0.2)
    print('[ICDD] after merging XRD-duplicate entries', len(precess.entries))

    print(len([_ for _ in precess.entries if _.structure.is_ordered]), 'ordered structures')
    print(len([_ for _ in precess.entries if not _.structure.is_ordered]), 'disordered structures')
    print(len([_ for _ in precess.entries if _.structure.composition.as_dict().keys() == {'V', 'O'}]))



    all_entries = precess.entries
    all_entries.append(sp) # This is right Mn2V2O7 beta phase

    df = get_dataframe([_ for _ in all_entries ],
                       ['entry_id', 'name', 'pressure_temperature', 'cross_refs', 'status', 'quality_mark', 'name',
                        'spgr', 'common_name', 'leader'])
    print(df)
    df.to_excel("output_candidate_pool.xlsx")

    with open('../data/icdd_entries.json', 'w') as f:
        json.dump(all_entries, f, cls=MontyEncoder)


    df = get_dataframe([_ for _ in all_entries if _.name == 'Mn2V2O7'],
                       ['entry_id', 'name', 'pressure_temperature', 'cross_refs', 'status', 'quality_mark', 'name',
                        'spgr', 'common_name', 'leader'])
    print(df) # check we get Mn2V2O7 phase correct)

if __name__ == "__main__":
    main()

