import os
import re
import xml.etree.cElementTree as ET
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.analysis.structure_matcher import StructureMatcher, ElementComparator
from pymatgen.core import Element, Composition
from scipy.cluster.hierarchy import fcluster, linkage
from monty.json import MSONable
from pymatgen.io.cif import CifParser
from scipy.ndimage import gaussian_filter1d


SITE_OCC_TOL = 0.15
COMP_TOL = 0.1
pd.options.display.width = 1000
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


class ICDDEntry(MSONable):
    def __init__(self, entry_id, empirical_formula, status, quality_mark, pressure_temperature, database_comments,
                 spgr, common_name, cross_refs, structure, name=None, leader=None, data=None):
        # # composition,
        # name, atomic_coords, has_struct, struct, ):
        self.entry_id = entry_id
        self.empirical_formula = empirical_formula
        self.composition = Composition(empirical_formula)
        self.status = status
        self.quality_mark = quality_mark
        self.pressure_temperature = pressure_temperature
        self.database_comments = database_comments
        self.spgr = spgr
        self.name = name if name else self.composition.reduced_formula
        self.common_name = common_name
        self.cross_refs = cross_refs
        self.structure = structure
        self.data = data if data else {}
        self.leader = leader if leader else self.entry_id

    @classmethod
    def from_icdd_xml(cls, xmlfile):
        root = ET.parse(xmlfile).getroot()[0]

        labels = ['pdf_number', 'empirical_formula', 'status', 'quality_mark', 'pressure_temperature',
                  'database_comments',
                  'spgr', 'common_name', 'cross_ref_pdf_numbers']

        def get_xml_value(root, key):
            el = root.findall(key)
            return el[0].text if el else ''

        params = [get_xml_value(root, label) for label in labels]

        index = labels.index('cross_ref_pdf_numbers')
        former = get_xml_value(root, 'former_pdf_number')
        if former:
            params[index] = ', '.join(params[index].split(', ') + [f"{_} (Former)" for _ in former.split(', ')])
            if params[index][0] == ',':
                params[index] = params[index][2:]
        params[index] = re.findall('\d{2}-\d{3}-\d{4}', params[index])

        # atomic_coords_el = root[0].findall('atomic_coords')
        # atomic_coords = 'Y' if atomic_coords_el else 'N'
        try_cif = '.'.join(xmlfile.split('.')[:-1]) + '.cif'
        if os.path.exists(try_cif):
            struct = CifParser(try_cif, occupancy_tolerance=1.1).get_structures(primitive=False)[0]
        else:
            struct = None

        return cls(*params, struct)

    @property
    def has_icsd_ref(self):
        if re.findall('ICSD Collection Code: [0-9]+', self.database_comments):
            return True
        # elif re.findall('01-\d{3}-\d{4} \(Former\)', self.cross_ref):
        #     return True
        else:
            return False
        # return ('ICSD' in self.database_comments)

    @property
    def former_icsd_ref(self):
        return re.findall('01-\d{3}-\d{4} \(Former\)', self.cross_refs)

    @property
    def qm_rank(self):
        return {'Star': 2, 'Indexed': 1, 'Prototyping': 1, 'Calculated': 1, 'Hypothetical': 0, 'Blank': 0,
                'Low-Precision': 0}[self.quality_mark]

    @property
    def status_rank(self):
        return {'Primary': 2, 'Alternate': 1, 'Deleted': 0}[self.status]

    @property
    def rank(self):
        return (100 * (self.structure.is_ordered) + self.status_rank * 1 + self.qm_rank * 10, self.entry_id)

    @property
    def icsd_id(self):
        icsd_entry_id = ''
        if self.has_icsd_ref:
            icsd_label = re.findall('ICSD Collection Code: [0-9]+', self.database_comments)
            icsd_entry_id = int(icsd_label[0].split(':')[-1])
        return icsd_entry_id

    def __eq__(self, other):
        return self.entry_id == other.entry_id

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self.entry_id)


class ICDDEntryPreprocessor:
    def __init__(self, all_entries, chemsys, oxide_system):
        self.all_entries = all_entries
        self.chemsys = chemsys
        self.oxide_system = oxide_system

    @property
    def entries(self):
        return [_ for _ in self.all_entries if _.leader == _.entry_id]

    def merge_by_cross_ref(self):

        n = len(self.all_entries)
        nn_matrix = np.ones([n, n])
        entry_ids = [_.entry_id for _ in self.entries]

        for i, e in enumerate(self.all_entries):
            for sibling in e.cross_refs:
                try:
                    j = entry_ids.index(sibling)
                    nn_matrix[i, j] = 0
                    nn_matrix[j, i] = 0
                except ValueError as e:
                    pass

        link = linkage(nn_matrix[np.triu_indices(n, k=1)])
        clusters = fcluster(link, 0.5, criterion='distance')
        clusters = clusters - 1

        groups = [[] for _ in set(clusters)]

        for i, e in zip(clusters, self.entries):
            groups[i].append(e)
        for g in groups:
            g.sort(key=lambda x: x.rank, reverse=True)

        for group in groups:
            for e in group:
                e.leader = group[0].entry_id

    def process_disorder(self):
        global SITE_OCC_TOL
        for e in [_ for _ in self.entries if not _.structure.is_ordered]:
            remove_index = []
            for i, site in enumerate(e.structure):
                if site.is_ordered:
                    continue
                if max(site.species.values()) <= SITE_OCC_TOL:
                    remove_index.append(i)
                    continue
                for el, amt in site.species.items():
                    if amt >= 1 - SITE_OCC_TOL:
                        site.species = el
                        break
            e.structure.remove_sites(remove_index)
            e.composition = e.structure.composition
            if '.' in e.name and ('.' not in e.structure.composition.reduced_formula):
                e.name = e.structure.composition.reduced_formula
        if self.oxide_system:
            exclude_entry_indexes = [_ for _ in self.all_entries if e.composition[Element('O')] <= 1e-4]
            self.all_entries = [e for e in self.all_entries if e.composition[Element('O')] > 0]

    def process_frac_name(self):
        multiplier = np.array([1, 2, 3, 4, 5, 6, 11])
        global COMP_TOL

        def reduce_comp(composition, cutoff=50):
            if sum(composition.values()) <= cutoff:
                return composition
            comps = [composition + Composition({el: 1}) for el in composition.keys()]
            comps += [composition - Composition({el: 1}) for el in composition.keys()]
            comps.append(composition)
            comps = [_.reduced_composition for _ in comps]
            comps.sort(key=lambda x: sum(x.values()))
            return comps[0]

        def get_interger_name(composition):
            els = list(composition.keys())
            amts = [composition[el] for el in els]
            amts = np.array(amts)
            frac = np.abs(amts - np.around(amts))
            if max(frac) <= COMP_TOL:
                amts = np.around(amts)
                return Composition.from_dict({el: amt for el, amt in zip(els, amts)})

            prod = np.array([list(composition.values())]) * multiplier.reshape(-1, 1)
            residual = np.max(np.abs(prod - np.around(prod)), axis=1) / multiplier
            if np.min(residual) <= COMP_TOL:
                multi = multiplier[np.argmin(residual)]
                comp_dict = composition.as_dict()
                for el, amt in comp_dict.items():
                    comp_dict[el] = np.around(amt * multi)
                composition = Composition.from_dict(comp_dict).reduced_composition

                while reduce_comp(composition) != composition:
                    composition = reduce_comp(composition)
            return composition.reduced_composition

        for e in self.entries:
            if '.' in e.name:
                comp1 = get_interger_name(e.composition)
                e.composition = comp1.reduced_composition
                e.name = e.composition.reduced_formula

    def get_xrd(self, leader_only=True):
        if leader_only:
            entries = self.entries
        else:
            entries = self.all_entries

        for e in entries:
            xrdcal = XRDCalculator()
            s = e.structure
            xrd = xrdcal.get_pattern(s, scaled=False)
            d = np.array(xrd.d_hkls)
            amplitude = np.array(xrd.y) / s.volume ** 2  # This is irrelevant of volume
            q_vectors = 4 * np.pi / (2 * d) * 10
            data = (q_vectors, amplitude)
            e.data['xrd'] = data

    def merge_by_structure(self, plot=False):
        sm = StructureMatcher(comparator=ElementComparator(), attempt_supercell=True)
        names = set([_.name for _ in self.entries])
        name_dict = {name: [_ for _ in self.entries if _.name == name] for name in names}

        for name, polymorphs in name_dict.items():
            if len(polymorphs) > 1:
                struct_match_tri = []
                for i, j in combinations(range(len(polymorphs)), 2):
                    e1, e2 = polymorphs[i], polymorphs[j]
                    s1, s2 = e1.structure, e2.structure
                    if 0.9 < (s1.volume / len(s1)) / (s2.volume / len(s2)) < 1.1:
                        if sm.fit(s1, s2):
                            struct_match_tri.append(0)
                        else:
                            struct_match_tri.append(1)
                    else:
                        struct_match_tri.append(1)
                links = linkage(struct_match_tri)
                clusters = fcluster(links, 0.5, 'distance')

                for e, c in zip(polymorphs, clusters):
                    e.color = f'C{c - 1}'
                polymorphs.sort(key=lambda x: (x.color, x.rank, x.entry_id), reverse=True)

                if plot:
                    # xrdcal=XRDCalculator()
                    print(name, len(polymorphs), len(set(clusters)), [_.entry_id for _ in polymorphs])
                    fig, axes = plt.subplots(ncols=1, nrows=len(polymorphs), sharex=True)
                    for ax, e in zip(axes, polymorphs):
                        # p = xrdcal.get_pattern(e.structure)
                        ax.stem(*e.data['xrd'], use_line_collection=True, markerfmt=" ", linefmt=e.color,
                                label=e.entry_id)
                        ax.legend(loc=1)
                    plt.show()

                color = polymorphs[0].color
                leader = polymorphs[0].entry_id
                for e in polymorphs[1:]:
                    if e.color == color:
                        for oe in self.all_entries:
                            if oe.leader == e.entry_id:
                                oe.leader = leader
                    else:
                        color = e.color
                        leader = e.entry_id

    def merge_by_xrd(self, bin_number, gaussian_filter, R_cutoff, plot=False):
        names = set([_.name for _ in self.entries])
        name_dict = {name: [_ for _ in self.entries if _.name == name] for name in names}

        def smooth_hist(q, amp, bins):
            hist, bin_edges = np.histogram(q, bins=bins, weights=amp)
            smoothed = gaussian_filter1d(hist, gaussian_filter)
            return smoothed

        for name, polymorphs in name_dict.items():
            if len(polymorphs) > 1:
                bins = np.linspace(min([_.data['xrd'][0][0] for _ in polymorphs]) - 0.01,
                                   max([_.data['xrd'][0][-1] for _ in polymorphs]) + 0.01, bin_number)
                xrd_match_tri = []
                for i, j in combinations(range(len(polymorphs)), 2):
                    e1, e2 = polymorphs[i], polymorphs[j]
                    s1, s2 = e1.structure, e2.structure
                    if 0.9 < (s1.volume / len(s1)) / (s2.volume / len(s2)) < 1.1:
                        ratio = (s1.volume / len(s1) / (s2.volume / len(s2))) ** (1 / 3)
                        q, amp = e1.data['xrd'][0], e1.data['xrd'][1]
                        smooth_xrds_i = smooth_hist(q, amp, bins)
                        q, amp = e2.data['xrd'][0], e2.data['xrd'][1]
                        smooth_xrds_j = smooth_hist(q / ratio, amp, bins)
                        smooth_xrds_i = smooth_xrds_i / np.max(smooth_xrds_i) * 100
                        smooth_xrds_j = smooth_xrds_j / np.max(smooth_xrds_j) * 100
                        abs_diff = np.abs(smooth_xrds_i - smooth_xrds_j)
                        R = np.sqrt(np.sum(abs_diff ** 2) / max(np.sum(smooth_xrds_i ** 2), np.sum(smooth_xrds_j ** 2)))
                        match = 0 if R < R_cutoff else 1
                    else:
                        match = 1
                    xrd_match_tri.append(match)
                links = linkage(xrd_match_tri)
                clusters = fcluster(links, 0.5, 'distance')

                for e, c in zip(polymorphs, clusters):
                    e.color = f'C{c - 1}'
                polymorphs.sort(key=lambda x: (x.color, x.rank, x.entry_id), reverse=True)

                if plot:
                    print(name, len(polymorphs), len(set(clusters)), [_.entry_id for _ in polymorphs])
                    fig, axes = plt.subplots(ncols=1, nrows=len(polymorphs), sharex=True)
                    for ax, e in zip(axes, polymorphs):
                        ax.stem(*e.data['xrd'], use_line_collection=True, markerfmt=" ", linefmt=e.color,
                                label=e.entry_id)
                        ax.legend(loc=1)
                    plt.show()

                color = polymorphs[0].color
                leader = polymorphs[0].entry_id
                for e in polymorphs[1:]:
                    if e.color == color:
                        for oe in self.all_entries:
                            if oe.leader == e.entry_id:
                                oe.leader = leader
                    else:
                        color = e.color
                        leader = e.entry_id
# added by Dongfang Yu  to get the Bi-Cu-V ICDDEntries
class ICDDEntriesBiCuV(ICDDEntry):
    def __init__ (self,common_name, composition, data, formula, entry_id, name, q, amp, crystal_system):
        self.common_name = common_name
        self.comp = composition
        self.data = data
        self.formula = formula
        self.composition = Composition(formula)
        self.entry_id = entry_id
        self.name = name
        self.q = q
        self.amp = amp
        self.crystal_system = crystal_system
    @classmethod
    def from_ICDD_Bi_Cu_V(cls, entry):
        data = {}
        data['xrd'] = [np.array(entry['entries_info']['q']), np.array(entry['entries_info']['amp'])]
        return cls(entry['entries_info']['name'],entry['entries_info']['comp'],
                   data,entry['entries_info'][ 'name'],
                   entry['entries_info']['entry_id'],entry['entries_info']['name'],entry['entries_info']['q'],
                   entry['entries_info']['amp'],entry['entries_info']['crystal_system'])