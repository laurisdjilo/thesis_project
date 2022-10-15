import pandas as pd
import random

DATA_PATH = "./data"

alternates_pairs = pd.read_csv(DATA_PATH+"/all_alternates_pairs.csv")
all_parts_set = set(alternates_pairs['original_mpn']).union(alternates_pairs['alternate_mpn'])

triplet_loss_dataset = pd.DataFrame(columns=['anchor', 'positive', 'negative'])
for i, pair in alternates_pairs.iterrows():
    non_alternates_sample = random.choices(tuple(all_parts_set.difference(alternates_pairs[alternates_pairs['original_mpn']==pair['original_mpn']]['alternate_mpn']).difference({pair['original_mpn']})), k=100)
    triplet_loss_dataset = pd.concat([
        triplet_loss_dataset,
        pd.DataFrame(map(lambda part: {'anchor': pair['original_mpn'], 'positive': pair['alternate_mpn'] , 'negative': part}, non_alternates_sample))
    ])

triplet_loss_dataset.to_csv(DATA_PATH+"/triplet_loss_model_dataset/triplets_dataset.csv", index=False)
