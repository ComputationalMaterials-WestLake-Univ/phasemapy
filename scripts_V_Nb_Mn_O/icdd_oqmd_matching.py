import json
import sys

sys.path += ['/Users/yizhou/PycharmProjects/phasemapy/phasemapy']

import qmpy

from monty.json import MontyDecoder, MontyEncoder
import pandas as pd
from collections import defaultdict
from pymatgen.core import Lattice, Structure, PeriodicSite

with open('../data/icdd_entries.json') as f:
    all_entries = json.load(f, cls=MontyDecoder)
print(len(all_entries))


def pymatgen2oqmd_struct(s):
    s.to('poscar', 'temp.poscar')
    new_s = qmpy.io.poscar.read('temp.poscar')
    new_s.make_primitive()
    return new_s


def oqmd2pymatgen_struct(d):
    sites = []
    lat = Lattice(d['unit_cell'])
    for s in d['sites']:
        sp, _, x, y, z = s.split()
        x, y, z = map(float, [x, y, z])
        site = PeriodicSite(sp, [x, y, z], lat)
        sites.append(site)
    struct = Structure.from_sites(sites)
    return struct


is_ordered = defaultdict(lambda: None)
oqmd_status = defaultdict(lambda: None)
oqmd_id = defaultdict(lambda: None)
stability_oqmd = defaultdict(lambda: 1000)

for i, e in enumerate(all_entries):
    if not oqmd_id[e.entry_id]:
        print(i, e.entry_id, e.name, e.composition)
        if e.structure.is_ordered:
            is_ordered[e.entry_id] = True
            dup = qmpy.Entry.get(pymatgen2oqmd_struct(e.structure))
            if dup:
                print(dup)
                if dup.energy:
                    # print (dup.energy)
                    oqmd_id[e.entry_id] = dup.id
                else:
                    dups = list(dup.duplicates.all())
                    dups = [_ for _ in dups if _.energy]
                    dups.sort(key=lambda x: x.id)
                    if len(dups):
                        oqmd_id[e.entry_id] = dups[0].id
                        # print ([_.energy for _ in dups])
            else:
                path = f'/home/oqmd/libraries/yizhou/sara/{e.entry_id}'
                if qmpy.Entry.objects.filter(path=path).exists():
                    oqmd_id[e.entry_id] = qmpy.Entry.objects.get(path=path).id
        else:
            is_ordered[e.entry_id] = False
            path = f'/home/oqmd/libraries/yizhou/sara/disordered/{e.entry_id}_{e.name}'
            sub_entries = []
            for j in range(30):
                subpath = path + f'/{j}'
                if qmpy.Entry.objects.filter(path=subpath).exists():
                    sub_entries.append(list(qmpy.Entry.objects.filter(path=subpath))[0])
                sub_entries.sort(key=lambda x: x.energy)
            oqmd_id[e.entry_id] = [_.id for _ in sub_entries]

            pd.set_option('display.float_format', lambda x: '%.3f' % x)

            print_stability = []
            oqmd_status = defaultdict(lambda x: None)

            print_oqmd_id = []

            for e in all_entries:
                if e.structure.is_ordered:
                    print_oqmd_id.append(str(oqmd_id[e.entry_id]))
                    print_stability.append(stability_oqmd[e.entry_id])
                    # print_stability.append('{:.3f}'.format(stability_oqmd[e.entry_id]))
                    if oqmd_id[e.entry_id]:
                        st = 'New' if oqmd_id[e.entry_id] > 1430000 else 'Old'
                    else:
                        st = '-'
                else:
                    print_oqmd_id.append(str(oqmd_id[e.entry_id][0]))
                    # print_stability.append('{:.3f}'.format(min(stability_oqmd[e.entry_id])))
                    print_stability.append(min(stability_oqmd[e.entry_id]))
                    # print_stability.append(', '.join(['{:.3f}'.format(_) for _ in stability_oqmd[e.entry_id]]))
                    st = 'New'
                oqmd_status[e.entry_id] = st

                # print (print_stability)
            include_st = [(_ < 0.1) for _ in print_stability]

            df = pd.DataFrame(data={
                'ICDD_id': [_.entry_id for _ in all_entries],
                'name': [_.name for _ in all_entries],
                'Is ordered': [is_ordered[_.entry_id] for _ in all_entries],
                'OQMD id': print_oqmd_id,  # [str(oqmd_id[_.entry_id]) for _ in all_entries],
                'OQMD status': [str(oqmd_status[_.entry_id]) for _ in all_entries],
                'stability': print_stability,
                'include': include_st
                # 'stability':['{:.2f}'.format(stability_oqmd[_.entry_id]) for _ in all_entries],
            })

            new_df = df.sort_values(['OQMD status', 'Is ordered', 'name'])
            # new_df.reset_index(drop=True)
            new_df = new_df.reset_index(drop=True)
            print(new_df)

pd.set_option('display.float_format', lambda x: '%.3f' % x)

print_stability = []
oqmd_status = defaultdict(lambda x: None)

print_oqmd_id = []

for e in all_entries:
    if e.structure.is_ordered:
        print_oqmd_id.append(str(oqmd_id[e.entry_id]))
        print_stability.append(stability_oqmd[e.entry_id])
        # print_stability.append('{:.3f}'.format(stability_oqmd[e.entry_id]))
        if oqmd_id[e.entry_id]:
            st = 'New' if oqmd_id[e.entry_id] > 1430000 else 'Old'
        else:
            st = '-'
    else:
        print_oqmd_id.append(str(oqmd_id[e.entry_id][0]))
        # print_stability.append('{:.3f}'.format(min(stability_oqmd[e.entry_id])))
        print_stability.append(min(stability_oqmd[e.entry_id]))
        # print_stability.append(', '.join(['{:.3f}'.format(_) for _ in stability_oqmd[e.entry_id]]))
        st = 'New'
    oqmd_status[e.entry_id] = st

# print (print_stability)
include_st = [(_ < 0.1) for _ in print_stability]

df = pd.DataFrame(data={
    'ICDD_id': [_.entry_id for _ in all_entries],
    'name': [_.name for _ in all_entries],
    'Is ordered': [is_ordered[_.entry_id] for _ in all_entries],
    'OQMD id': print_oqmd_id,  # [str(oqmd_id[_.entry_id]) for _ in all_entries],
    'OQMD status': [str(oqmd_status[_.entry_id]) for _ in all_entries],
    'stability': print_stability,
    'include': include_st
    # 'stability':['{:.2f}'.format(stability_oqmd[_.entry_id]) for _ in all_entries],
})

new_df = df.sort_values(['OQMD status', 'Is ordered', 'name'])
# new_df.reset_index(drop=True)
new_df = new_df.reset_index(drop=True)
print(new_df)

new_df.to_csv('icdd_oqmd.csv')
prune_entries = []
for b, e in zip(include_st, all_entries):
    if b:
        prune_entries.append(e)
len(prune_entries)





with open('../data/icdd_oqmd_entries.json', 'w') as f:
    json.dump(prune_entries, f, cls=MontyEncoder)
len(prune_entries)

# from phasemapy.solver import Sample, Phase
