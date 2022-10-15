import pandas as pd
import random

DATA_PATH = "./data"

alternates_pairs = pd.read_csv(DATA_PATH+"/all_alternates_pairs.csv")
all_parts_set = set(alternates_pairs['original_mpn']).union(alternates_pairs['alternate_mpn'])

non_alternates_pairs = pd.DataFrame(columns=['original_mpn', 'alternate_mpn'])
for part in all_parts_set:
    sample_non_alternates = random.choices(tuple(all_parts_set.difference(alternates_pairs[alternates_pairs['original_mpn']==part]['alternate_mpn']).difference({part})), k=50)
    non_alternates_pairs = pd.concat([
        non_alternates_pairs,
        pd.DataFrame(map(lambda alt_part: {'original_mpn': part, 'alternate_mpn': alt_part}, sample_non_alternates))
    ])

alternates_pairs['similarity'] = 1
non_alternates_pairs['similarity'] = 0

pd.concat([alternates_pairs, non_alternates_pairs]).to_csv(DATA_PATH+'/cross_entropy_loss_dataset/pairs_dataset_50.csv', index=False)

