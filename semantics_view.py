import pandas as pd
import random
import math
import faiss
import lmdb
import os
import logging
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn
from math import sqrt
import datetime
import logging
from utils import *
import pickle
from InfoNCE.infonce import InfoNCE, info_nce
from st_moe_pytorch import MoE, SparseMoEBlock
from typing import Optional, Any, Union, Callable
import geohash
from haversine import Unit, haversine_vector

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class PretrainDataset(Dataset):
    def __init__(self, poi_df, data_df, mode):
        super().__init__()
        self.poi_df = poi_df
        self.data_df = data_df
        self.poi_index_list = poi_df.index.tolist()
        self.poi_count = len(poi_df)
        self.mode = mode
        self.neg_count = NEG_COUNT
        self.replay_memory = None
        if mode == "train":
            self.txn = train_txn
        elif mode == "eval":
            self.txn = eval_txn
        elif mode == "test":
            self.txn = test_txn
    
    def __len__(self):
        return len(self.data_df)

    def update_replay_memory(self, replay_memory):
        self.replay_memory = replay_memory

    def __getitem__(self, index):
        query_data = self.data_df.iloc[index]
        pos_id = query_data["pos_id"]
        pos_data = self.poi_df.loc[pos_id]
        query_text = torch.FloatTensor(pickle.loads(self.txn.get(str(index).encode())))
        query_geo_code = query_data[geo_code_type]
        query_geo_code = torch.LongTensor([hash2index[hash_] for hash_ in start_token + query_geo_code])
        
        pos_geo_code =  pos_data[geo_code_type]
        pos_geo_code = torch.LongTensor([hash2index[hash_] for hash_ in start_token + pos_geo_code])
        
        if self.mode == "train":
            neg_set = set()
            if self.replay_memory:
                memory = self.replay_memory[index]
                dense_memory, geo_prob, geo_distance = memory["dense"], memory["geo_prob"], memory["distance"]
                geo_memory_set = set(np.random.choice(dense_memory, size=GEO_MEMORY_COUNT+1, replace=False, p=geo_prob))
                geo_memory_list = list(geo_memory_set - set([pos_id]))[:GEO_MEMORY_COUNT]
                neg_set.update(geo_memory_list)
                for i in range(len(dense_memory)):
                    if len(neg_set) == MEMORY_COUNT: break
                    neg_set.add(dense_memory[i])
                neg_set = neg_set - set([pos_id])

            while len(neg_set) < self.neg_count:
                random_index = random.sample(self.poi_index_list, 1)[0]
                if random_index != pos_id:
                    neg_set.add(random_index)
            neg_list = list(neg_set)
            random.shuffle(neg_list)

            neg_address = torch.FloatTensor([pickle.loads(poi_txn.get(str(neg_id).encode())) for neg_id in neg_list])
            neg_df = self.poi_df.loc[neg_list]
            neg_geo_code = neg_df[geo_code_type]
            neg_geo_code = torch.LongTensor([[hash2index[hash_] for hash_ in start_token + neg_geo] for neg_geo in neg_geo_code])

            pos_address = torch.FloatTensor(pickle.loads(poi_txn.get(str(pos_id).encode())))
            
            data = {
                "query_text": query_text, "query_geo_code": query_geo_code,
                "pos_address": pos_address, "pos_geo_code": pos_geo_code,
                "neg_address": neg_address, "neg_geo_code": neg_geo_code,
            }
        elif self.mode == "test" or self.mode == "eval":
            data = {
                "query_text": query_text, "query_geo_code": query_geo_code,
                "pos_geo_code": pos_geo_code, "pos_id": torch.LongTensor([pos_id])
            }
        return data


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=200):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = self.dropout(x + self.pe[:, :x.size(1), :])
        return x

class Encoder(nn.Module):
    def __init__(self, pretrain_dim, model_dim, device):
        super().__init__()
        self.dim = model_dim
        self.geo_char_len = len(geo_code_char)

        self.pos_encoder = PositionalEncoding(self.dim)

        self.text_encoder = nn.Sequential(
            nn.LayerNorm(pretrain_dim),
            nn.Linear(pretrain_dim, pretrain_dim),
            nn.LeakyReLU(),
            nn.Linear(pretrain_dim, self.dim),
        )

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.dim, nhead=2, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        
        merge_moe = MoE(
            dim = self.dim,
            num_experts = 2,               # increase the experts (# parameters) of your model without increasing computation
            gating_top_n = 2,               # default to top 2 gating, but can also be more (3 was tested in the paper with a lower threshold)
            threshold_train = 0.2,          # at what threshold to accept a token to be routed to second expert and beyond - 0.2 was optimal for 2 expert routing, and apparently should be lower for 3
            threshold_eval = 0.2,
            capacity_factor_train = 1.25,   # experts have fixed capacity per batch. we need some extra capacity in case gating is not perfectly balanced.
            capacity_factor_eval = 2.,      # capacity_factor_* should be set to a value >=1
            balance_loss_coef = 1e-2,       # multiplier on the auxiliary expert balancing auxiliary loss
            router_z_loss_coef = 1e-3,      # loss weight for router z-loss
        )
        self.merge_moe_block = SparseMoEBlock(
            merge_moe,
            add_ff_before = True,
            add_ff_after = True
        )

        self.merge_attention = nn.MultiheadAttention(embed_dim=self.dim, num_heads=1, batch_first=True)

        self.geohash_embedding = nn.Embedding(num_embeddings=self.geo_char_len, embedding_dim=self.dim)

        self.attention_pooling = nn.Sequential(
            nn.Linear(self.dim, 1)
        )
        self.attention_layer_norm = nn.LayerNorm(self.dim)
        
        self.feature_merge = nn.Sequential(
            nn.Linear(self.dim * 2 , self.dim),
            nn.LayerNorm(self.dim),
            nn.ReLU()
        )

        self.sim_block = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.dim * 2),
            nn.LeakyReLU(),
            nn.LayerNorm(self.dim * 2),
            nn.Linear(self.dim * 2, self.dim * 4),
            nn.LeakyReLU(),
            nn.LayerNorm(self.dim * 4),
            nn.Linear(self.dim * 4, self.dim * 2),
            nn.LeakyReLU(),
            nn.LayerNorm(self.dim * 2),
            nn.Linear(self.dim * 2, self.dim),
        )

        self.sigma = torch.nn.Parameter(torch.ones(2, device=device))
        

    def get_geo_embedding(self, geo_code, mode):
        geo_embedding = self.geohash_embedding(geo_code)
        return geo_embedding
    
    def get_text_embedding(self, text_embedding, mode):
        text_embedding = self.text_encoder(text_embedding)
        return text_embedding
    
    def get_cross_modal_embedding(self, text_embedding, geo_code, mode):
        b_size = text_embedding.size(0)
        if mode == "neg":
            text_embedding = text_embedding.view(-1, text_embedding.size(-1))
            geo_code = geo_code.view(-1, geo_code.size(-1))
        
        text_embedding = self.text_encoder(text_embedding)
        geo_embedding = self.geohash_embedding(geo_code)
        
        cat_embedding = torch.cat([text_embedding.unsqueeze(1), geo_embedding], dim=1)
        cat_embedding, aux_loss, _, _ = self.merge_moe_block(cat_embedding)
        cross_text_embedding, _attn_weight = self.merge_attention(text_embedding.unsqueeze(1), cat_embedding, cat_embedding)
        if mode == "neg":
            cross_text_embedding = cross_text_embedding.view(b_size, -1, self.dim)
        cross_text_embedding = self.sim_block(cross_text_embedding)
        return {"cross_text_embedding": cross_text_embedding}
        

    def forward(self, text_embedding, geo_code, mode):
        text_embedding = self.get_text_embedding(text_embedding, mode)
        geo_embedding = self.get_geo_embedding(geo_code, mode)
        
        return {"text_embedding": text_embedding, "geo_embedding": geo_embedding}

class PretrainModel(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        
        self.pretrain_dim = 1024
        self.dim = 256
        self.encoder = Encoder(self.pretrain_dim, self.dim, device)

        self.decoder_layer = nn.TransformerDecoderLayer(d_model=self.dim, nhead=4, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=4)

        self.geo_char_len = len(geo_code_char)
        self.vocab_size = self.geo_char_len - 1
        self.predict_layer = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.dim * 2),
            nn.LeakyReLU(),
            torch.nn.Linear(self.dim * 2, self.vocab_size)
        )

        self.geo_decoder = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.dim),
            nn.LeakyReLU(),
            nn.Linear(self.dim, len(geo_code_char)),
        )

        
    def forward(self, data, stage):
        query_cross_text_embedding = self.encoder.get_cross_modal_embedding(data["query_text"], data["query_geo_code"], mode="query")["cross_text_embedding"].squeeze(1)

        if stage == "train":
            pos_cross_text_embedding = self.encoder.get_cross_modal_embedding(data["pos_address"], data["pos_geo_code"], mode="pos")["cross_text_embedding"].squeeze(1)
            neg_cross_text_embedding = self.encoder.get_cross_modal_embedding(data["neg_address"], data["neg_geo_code"], mode="neg")["cross_text_embedding"]

            return {"query_embedding": query_cross_text_embedding, "pos_embedding": pos_cross_text_embedding, "neg_embedding": neg_cross_text_embedding}
        elif stage == "test":
            return {"query_embedding": query_cross_text_embedding}
    
class OursLossFunction(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.infonce = InfoNCE()
        self.device = device

    def forward(self, outputs, data):
        query_embedding, pos_embedding, neg_embedding = outputs["query_embedding"], outputs["pos_embedding"], outputs["neg_embedding"]
        loss = self.infonce(query_embedding, pos_embedding, neg_embedding.view(-1, neg_embedding.size(-1)))
        
        return {"loss": loss}

class Trainner:
    def __init__(self):
        self.device = torch.device(f"cuda:{device_idx}" if torch.cuda.is_available() else "cpu")
        self.model = PretrainModel(self.device).to(self.device)
        self.our_loss_fn = OursLossFunction(self.device)
        self.optimizer = torch.optim.AdamW(list(self.model.parameters()) + list(self.our_loss_fn.parameters()), lr=1e-4)
        self.scheduler = torch.optim.lr_scheduler.StepLR(optimizer=self.optimizer, step_size=4, gamma=0.8)
        self.scaler = torch.amp.GradScaler()
        self.save_dir = "./save/checkpoints"
        self.max_keep = 5
        self.K_list = [1, 3, 5, 10]
        self.correct_dict = {K: 0 for K in self.K_list}
        self.test_vec_db = None
        self.gpu_index = None
        self.docid2index = {doc_id: index for index, doc_id in enumerate(poi_df["doc_id"])}
        
        self.train_dataset = PretrainDataset(poi_df=poi_df, data_df=train_df, mode="train")
        self.train_loader = DataLoader(self.train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
        
        self.eval_dataset = PretrainDataset(poi_df=poi_df, data_df=eval_df, mode="eval")
        self.eval_loader = DataLoader(self.eval_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
        
        self.test_dataset = PretrainDataset(poi_df=poi_df, data_df=test_df, mode="test")
        self.test_loader = DataLoader(self.test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
        
    def train(self, epoch):
        self.model.train()
        result = self.iteration(epoch, self.train_loader, mode="train")
        self.scheduler.step()
        return result
    
    def eval(self, epoch):
        self.model.eval()
        result = self.iteration(epoch, self.eval_loader, mode="eval")
        return result
    
    def test(self, epoch):
        self.model.eval()
        result = self.iteration(epoch, self.test_loader, mode="test")
        return result
    
    def trans_data(self, data, mode):
        for key, value in data.items():
            data[key] = value.to(self.device)
        return data
    
    def metrics(self, outputs, data):
        dense_correct = {K: 0 for K in K_list}
        pos_id = data["pos_id"].cpu()
        query_embedding = F.normalize(outputs["query_embedding"], dim=-1).cpu()
        search_distance, search_indices = self.gpu_index.search(query_embedding, max(K_list))
        mrr = 1 / (torch.where((pos_id == torch.tensor(search_indices[:, :max(K_list)])))[1] + 1)
        for K in K_list:
            dense_correct[K] += (pos_id == torch.tensor(search_indices[:, :K])).sum().item()

        return {"mrr": mrr.tolist(), "dense_correct": dense_correct}
        

    def iteration(self, epoch, data_loader, mode):
        epoch_loss_list = []
        cross_loss_list = []
        mrr_list = []
        dense_correct = {K: 0 for K in K_list}
        total = 0
        for index, data in enumerate(tqdm(data_loader, desc=f"{mode}")):
            data = self.trans_data(data, mode)
            if mode == "train":
                outputs = self.model(data, stage="train")
                loss_dict = self.our_loss_fn(outputs, data)
                loss = loss_dict["loss"]
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                epoch_loss_list.append(loss.item())

            if mode == "test" or mode == "eval":
                with torch.no_grad():
                    outputs = self.model(data, stage="test")
                total += outputs["query_embedding"].size(0)
                metric_results = self.metrics(outputs, data)
                mrr_list.extend(metric_results["mrr"])
                for k, correct_count in metric_results["dense_correct"].items():
                    dense_correct[k] += correct_count
                
        if mode == "train":
            logging.info(f"{epoch} | train loss: {np.mean(epoch_loss_list)}")
            return {"loss": np.mean(epoch_loss_list)}
        elif mode == "test" or mode == "eval":
            dense_correct_dict = {k: count / total * 100 for k, count in dense_correct.items()}
            logging.info(f"{mode} | correct: {dense_correct_dict}")
            logging.info(f"{mode} | mrr: {(sum(mrr_list) / total):.6f}")
            return {"dense_correct_dict": dense_correct_dict}
    

    def get_doc_vec(self, doc_embedding, geo_code_list):
        geo_tokens = [[hash2index[hash_] for hash_ in start_token + geo] for geo in geo_code_list]
        geo_tokens = torch.LongTensor(geo_tokens).to(self.device)

        doc_vec = self.model.encoder.get_cross_modal_embedding(doc_embedding, geo_tokens, mode="doc")["cross_text_embedding"].squeeze(1)
        return {"doc_vec": doc_vec}
    
    def build_faiss(self):
        poi_index = None
        batch_size = 2000
        poi_df_len = len(poi_df)
        for i in tqdm(range(0, poi_df_len, batch_size), desc="faiss build"):
            doc_id_list = poi_df[i:i+batch_size]["doc_id"].values
            geo_code_list = poi_df[i:i+batch_size][geo_code_type].tolist()
            index_list = poi_df[i:i+batch_size].index.tolist()

            doc_embedding = torch.FloatTensor([pickle.loads(poi_txn.get(str(index).encode())) for index in index_list]).to(self.device)
            
            with torch.no_grad():
                doc_vec = self.get_doc_vec(doc_embedding, geo_code_list)["doc_vec"]
                doc_vec = F.normalize(doc_vec, dim=-1).cpu()
            if poi_index is None:
                poi_index = faiss.IndexFlatIP(doc_vec.shape[-1])
                poi_index = faiss.IndexIDMap(poi_index)
            poi_index.add_with_ids(doc_vec, doc_id_list)
        self.gpu_index = poi_index

    def build_memory(self):
        batch_size = 1000
        memory_dict = dict()
        for i in tqdm(range(0, train_df.shape[0], batch_size), desc="build memory"):
            geo_code_list = train_df[i:i+batch_size][geo_code_type].tolist()
            index_list = train_df[i:i+batch_size].index.tolist()
            query_embedding = torch.FloatTensor([pickle.loads(train_txn.get(str(index).encode())) for index in index_list]).to(self.device)
            with torch.no_grad():
                query_vec = self.get_doc_vec(query_embedding, geo_code_list)["doc_vec"]
                query_vec = F.normalize(query_vec, dim=-1).cpu()

            _, search_indices = self.gpu_index.search(query_vec, 500)

            for index, dense in zip(index_list, search_indices):
                dense_geo = poi_df.iloc[dense][["lat", "lng"]].values
                memory_geo = geohash.decode(geo_memory[index]["geo_code"])
                distance = haversine_vector(memory_geo, dense_geo, Unit.KILOMETERS, comb=True).squeeze(-1)
                distance_score = 1 - np.tanh((distance - distance.min()) / (distance.max() - distance.min()))
                
                distance_prob = distance_score / distance_score.sum()
                memory_dict[index] = {"dense": dense, "geo_prob": distance_prob, "distance": distance.tolist()}
        self.train_dataset.update_replay_memory(memory_dict)

    def load_faiss(self):
        faiss_path = f"./save/v{v}_best_neg{NEG_COUNT}.index"
        if not os.path.exists(faiss_path):
            self.build_faiss()
            faiss.write_index(self.gpu_index, faiss_path)
        else:
            self.gpu_index = faiss.read_index(faiss_path)
    
    def generate(self):
        result = dict()
        is_list = False
        if is_list:
            K_max_list = [100, 150, 200, 250, 300, 350, 400, 450, 500]
        else:
            K_max_list = [400]
        for K_max in K_max_list:
            for i, data in enumerate(tqdm(self.test_loader, desc=f"generate")):
                data = self.trans_data(data, "generate")
                outputs = self.model(data, stage="test")
                query_embedding = outputs["query_embedding"]

                pos_ids = data["pos_id"].cpu().squeeze(1).tolist()
                query_embedding = F.normalize(query_embedding, dim=-1).detach().cpu()
                search_distance, search_indices = self.gpu_index.search(query_embedding, K_max)
                index_list = data["index"].tolist()
                for index, distance, indices, pos_id in zip(index_list, search_distance.tolist(), search_indices.tolist(), pos_ids):
                    result[index] = {"distance": distance, "indices": indices, "pos_id": pos_id}
                    pass
            np.save(f"./save/v{v}_dense_K{K_max}_neg_{NEG_COUNT}.npy", arr=result)

    def start(self):
        
        best_acc = 0
        epochs = 50
        for epoch in range(epochs):
            train_result = self.train(epoch)  # train
            self.build_faiss()
            self.build_memory()
            eval_result = self.eval(epoch)
            test_result = self.test(epoch)
            eval_acc = eval_result["dense_correct_dict"][10]
            if eval_acc > best_acc:
                best_acc = eval_acc
                checkpoint = {
                    "epoch": epoch, 
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "scheduler": self.scheduler.state_dict(),
                    "loss_fn": self.our_loss_fn.state_dict(),
                }
                torch.save(checkpoint, f"./save/checkpoint/v{v}_neg_count{NEG_COUNT}.pth")

def dt_converter(*args):
    now = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=8)
    return now.timetuple()


if __name__ == '__main__':
    v=14

    dataset_name = "geoglue"
    device_idx = 1
    random_seed = 0
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.device_count() > 1:
        torch.cuda.manual_seed_all(random_seed)
    else:
        torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    logging_format = "%(asctime)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    logging.Formatter.converter = dt_converter
    logging.basicConfig(filename=f"./save/pretrain_{v}.log", level=logging.INFO, format=logging_format, datefmt=date_format)

    BATCH_SIZE = 100
    NUM_WORKERS = 10
    MAX_LEN = 100
    NEG_COUNT = 150
    MEMORY_COUNT = NEG_COUNT // 2
    GEO_MEMORY_COUNT = NEG_COUNT // 4

    poi_env = lmdb.open("../recall_code/save/embedding/poi", readonly=True)
    poi_txn = poi_env.begin()
    train_env = lmdb.open("../recall_code/save/embedding/train", readonly=True)
    train_txn = train_env.begin()
    eval_env = lmdb.open("../recall_code/save/embedding/eval", readonly=True)
    eval_txn = eval_env.begin()
    test_env = lmdb.open("../recall_code/save/embedding/test", readonly=True)
    test_txn = test_env.begin()

    poi_df = pd.read_csv(f"../recall_data/{dataset_name}/dataset/poi.csv", header=0)
    poi_df["address"] = poi_df["address"].str.slice(0, MAX_LEN)
    train_df = pd.read_csv(f"../recall_data/{dataset_name}/dataset/train.csv", header=0)
    train_df["query"] = train_df["query"].str.slice(0, MAX_LEN)
    eval_df = pd.read_csv(f"../recall_data/{dataset_name}/dataset/eval.csv", header=0)
    eval_df["query"] = eval_df["query"].str.slice(0, MAX_LEN)
    test_df = pd.read_csv(f"../recall_data/{dataset_name}/dataset/test.csv", header=0)
    test_df["query"] = test_df["query"].str.slice(0, MAX_LEN)

    geo_memory = np.load("./save/geo_memory.npy", allow_pickle=True).tolist()

    trainner = Trainner()
    trainner.start()