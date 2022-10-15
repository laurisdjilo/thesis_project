import pandas as pd
import numpy as np
import networkx as nx

raw_alternate_parts = pd.read_csv("./data/alternate_parts_couples.csv")

def permutate(x):
    if x['original_mpn'] > x['alternate_mpn']:
        tmp = x['alternate_mpn']
        x['alternate_mpn'] = x['original_mpn']
        x['original_mpn'] = tmp
    return x

alternates_pairs = raw_alternate_parts.copy()
alternates_pairs = alternates_pairs.apply(permutate, axis=1)
alternates_pairs = alternates_pairs.drop_duplicates()

all_parts_set = np.sort(list(set(alternates_pairs['original_mpn']).union(alternates_pairs['alternate_mpn'])))

alternates_graph = nx.from_pandas_edgelist(alternates_pairs, 'original_mpn', 'alternate_mpn')

for i, x in enumerate(all_parts_set[:-1]):
    for y in set(all_parts_set[i+1:]).difference(alternates_pairs[alternates_pairs['original_mpn']==x]['alternate_mpn']).difference(alternates_pairs[alternates_pairs['alternate_mpn']==x]['original_mpn']):
        if nx.has_path(alternates_graph, x, y):
            alternates_pairs = pd.concat([
                alternates_pairs
                , pd.DataFrame([{'original_mpn': x, 'alternate_mpn': y}])
            ])

alternates_pairs.to_csv("./data/all_alternates_pairs.csv", index=False)
