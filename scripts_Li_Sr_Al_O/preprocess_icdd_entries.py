import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
import os
from copy import deepcopy
from scipy.constants import h, c, e
from monty.json import MontyDecoder, MontyEncoder
import json
import pandas as pd

from phasemapy.parser import ICDDEntry, ICDDEntryPreprocessor
from phasemapy.solver import Phase, Sample

chemsys = ['Li', 'Sr', 'Al']
oxide_system = True

def main():
    def get_dataframe(icdd_entries, keys):
        data = {}
        for key in keys:
            data[key] = [e.as_dict()[key] for e in icdd_entries]
        df = pd.DataFrame(data)
        return df


    def plot_merge_xrd(entries_sum, entries_index):
        from scipy.ndimage import gaussian_filter1d
        def smooth_hist(q, amp, bins):
            hist, bin_edges = np.histogram(q, bins=bins, weights=amp)
            smoothed = gaussian_filter1d(hist, 4)
            return smoothed

        bins = np.linspace(min([_.data['xrd'][0][0] for _ in entries_sum]) - 0.01,
                           max([_.data['xrd'][0][-1] for _ in entries_sum]) + 0.01, 1000)
        smooth_xrd_data = []
        for i in entries_index:
            q, amp = entries_sum[i].data['xrd'][0], entries[i].data['xrd'][1]
            smooth_xrds_i = smooth_hist(q, amp, bins)
            smooth_xrds_i = smooth_xrds_i / np.max(smooth_xrds_i) * 100
            smooth_xrd_data.append(smooth_xrds_i)

        for j in range(len(smooth_xrd_data)):
            plt.plot(bins[0:-1], smooth_xrd_data[j],
                     label=f"{entries_index[j]}+{entries_sum[entries_index[j]].name}+{entries_sum[entries_index[j]].entry_id}")
            plt.legend()

    # load entry pool: 100 ICDD entries
    with open('./data/ICDD_entries_raw.json') as f:
        entries_Li_Sr_Al = json.load(f, cls=MontyDecoder)

    entries = [ICDDEntry.from_icdd_json(en) for en in entries_Li_Sr_Al]
    precess = ICDDEntryPreprocessor(deepcopy(entries), chemsys, oxide_system)
    groups = precess.merge_by_xrd(bin_number=1000, gaussian_filter=4, R_cutoff=0.15)
    df = get_dataframe([_ for _ in precess.entries],
                       ['entry_id', 'name', 'leader'])
    # print(df)
    df.to_excel("./data/output_candidate_pool.xlsx")

    with open('./data/icdd_entries.json', 'w') as f:
        json.dump(precess.entries, f, cls=MontyEncoder)


if __name__ == "__main__":
    main()
