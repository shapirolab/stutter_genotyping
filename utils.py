from collections import namedtuple
import numpy as np
import pandas as pd

def extract_normalized_vectors(df3_40):
    vech_keys = []
    vech_values = []
    for k, r in df3_40.iterrows():
        vech = np.array(r)
        vech_normed = vech/np.linalg.norm(vech, ord=2)
        if np.any(np.isnan(vech_normed)):
            print('warning nans in {}'.format(k))
            continue
        vech_keys.append(k)
        vech_values.append(vech_normed)
    return vech_keys, vech_values

Call = namedtuple('Call',['hist','conf'])
def map_tree_query(ind, dist, vech_keys, hist_map):
    called = {}
    for (ind0, dist0, k) in zip(ind, dist, vech_keys):
        if len(ind0)!=0:
            imin = np.argmin(dist0)
            called[k] = Call(hist_map[ind0[imin]], dist_to_cosing_score(dist0[imin]))
    return called


import lzma
import pickle
def load_kd_tree(kd_tree_pickle_path="ac_kdtree.pickle", hist_map="ac_hist_map.pickle"):

    with lzma.open("kdtree.xz", "rb") as f:
        kdtree = pickle.load(f)


    with lzma.open("hist_map.xz", "rb") as f:
        hist_map = pickle.load(f)
    return kdtree, hist_map


def cosine_score_to_dist(alpha):
    return 2*np.sin(np.arccos(1-alpha)/2)

def dist_to_cosing_score(d):
    return 1-np.cos(2*np.arcsin(d/2))



from decimal import Decimal
conf = 0.05
CONF_RADIOS_THRESHOLD = cosine_score_to_dist(conf)

def genotype_stutter_histograms(df):
    print("extracting normalized vectors")
    vech_keys, vech_values = extract_normalized_vectors(df)
    print("loading reference histograms")
    kdtree, hist_map = load_kd_tree()
    print("querying kd-tree")
    (ind, dist) = kdtree.query_radius(vech_values, r=CONF_RADIOS_THRESHOLD, return_distance=True)
    print("mapping results")
    called = map_tree_query(ind, dist, vech_keys, hist_map)
    flat_called = dict()
    for k, call_item in called.items():
        alleles_and_proportions = sorted(call_item.hist.alleles_to_proportions.items())
        if len(alleles_and_proportions) == 1:
            alleles_and_proportions = alleles_and_proportions+[(0, Decimal('0.0000'))]
        flat_called[k] = list(alleles_and_proportions[0])+list(alleles_and_proportions[1])+[call_item.conf, call_item.hist.simulation_cycle]
    kd_calling_df = pd.DataFrame.from_dict(flat_called).T
    kd_calling_df.index.set_names(df.index.names, inplace=True)
    kd_calling_df.columns=['kd_a1', 'kd_prop1','kd_a2', 'kd_prop2', 'kd_conf', 'kd_cycles']
    return kd_calling_df