import pandas as pd
import numpy as np
from haversine import haversine, Unit, haversine_vector
from sklearn.neighbors import BallTree
from utils import *
import geohash
from tqdm import tqdm


dataset_name = "geoglue"
MAX_LEN = 100

poi_df = pd.read_csv(f"../recall_data/{dataset_name}/dataset/poi.csv", header=0)
poi_df["address"] = poi_df["address"].str.slice(0, MAX_LEN)
train_df = pd.read_csv(f"../recall_data/{dataset_name}/dataset/train.csv", header=0)
train_df["query"] = train_df["query"].str.slice(0, MAX_LEN)
val_df = pd.read_csv(f"../recall_data/{dataset_name}/dataset/eval.csv", header=0)
val_df["query"] = val_df["query"].str.slice(0, MAX_LEN)
test_df = pd.read_csv(f"../recall_data/{dataset_name}/dataset/test.csv", header=0)
test_df["query"] = test_df["query"].str.slice(0, MAX_LEN)

geo_ball_tree = BallTree(np.deg2rad(poi_df[["lat", "lng"]].values), leaf_size=100, metric="haversine")
km_trans = 1 / 6371.0088


v = 36
K = 400
NEG_COUNT = 150

dense_result = np.load(f"./save/v{v}_dense_K{K}_neg_{NEG_COUNT}.npy", allow_pickle=True).tolist()
geo_result = np.load(f"./save/geo_generate.npy", allow_pickle=True).tolist()

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def test():
    total = 0
    dense_correct, geo_correct, geo_full_correct, both_correct, all_correct = 0, 0, 0, 0, 0
    geo_top_100_correct = 0
    correct_dict = {K: 0 for K in K_list}
    mrr_list = []

    for i in tqdm(range(len(dense_result))):
        dense = dense_result[i]
        geo = geo_result[i]
        
        pos_id = dense["pos_id"]
        dense_pois = dense["indices"]
        dense_distance = dense["distance"]

        

        geo_code = geo["geo_code"]
        geo_code = "".join([index2hash[code] for code in geo_code[1:]])
        geo_lat, geo_lon = geohash.decode(geo_code)


        distance = haversine_vector([geo_lat, geo_lon], poi_df.iloc[dense_pois][["lat", "lng"]].values.tolist(), unit=Unit.KILOMETERS, comb=True).squeeze(1)

        distance_norm = 1 - np.tanh(np.arcsinh(distance))
        dense_norm = np.array(dense_distance)
        dense_norm = (dense_norm - dense_norm.min()) / (dense_norm.max() - dense_norm.min())
        merge_score = distance_norm + dense_norm
        merge_sort_index = merge_score.argsort()[::-1]

        merge_pois = np.array(dense_pois)[merge_sort_index].tolist()

        total += 1
        try:
            mrr = 1 / (merge_pois[:max(K_list)].index(pos_id) + 1)
        except Exception as e:
            mrr = 0
        mrr_list.append(mrr)

        for K in K_list:
            if pos_id in merge_pois[:K]:
                correct_dict[K] += 1
                
    print({key: value / total * 100 for key, value in correct_dict.items()})
    print(np.mean(mrr_list))

if __name__ == '__main__':
    test()