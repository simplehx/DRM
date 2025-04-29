import pandas as pd
import random
import math
import faiss
import os
from faiss import Index
import logging
import json
import torch
import torch.nn.functional as F
from tqdm import tqdm
from itertools import chain
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn
import datetime
import logging
from transformers.generation.beam_search import BeamHypotheses
from utils import *
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from InfoNCE.infonce import InfoNCE, info_nce
from st_moe_pytorch import MoE, SparseMoEBlock
from typing import Optional, Any, Union, Callable
import geohash
from haversine import haversine, Unit

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class ULPDataset(Dataset):
    def __init__(self, poi_df, data_df, mode):
        super().__init__()
        self.poi_df = poi_df
        self.data_df = data_df
        self.poi_index_list = poi_df.index.tolist()
        self.poi_count = len(poi_df)
        self.mode = mode
    
    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index):
        query_data = self.data_df.iloc[index]
        query = query_data["query"]
        pos_id = query_data["pos_id"]
        pos_data = self.poi_df.iloc[pos_id]
        pos_geo_code = pos_data[geo_code_type]
        
        if self.mode == "train":
            random_idx = random.randint(0, len(pos_geo_code)-1)
            mask = [0 if i==random_idx else 1 for i in range(len(pos_geo_code))]

            mask_geo_code = [hash2index[start_token]] + [hash2index[pos_geo_code[i]] if m == 1 else hash2index[mask_token] for i, m in enumerate(mask)]
            mask_ground_truth = [-100] + [-100 if m == 1 else hash2index[pos_geo_code[i]] for i, m in enumerate(mask)]
            pos_geo_code = torch.LongTensor([hash2index[hash_] for hash_ in start_token + pos_geo_code])
            mask_geo_code = torch.LongTensor(mask_geo_code)
            mask_ground_truth = torch.LongTensor(mask_ground_truth)
            data = {"query_text": query, "pos_geo_code": pos_geo_code, "mask_geo_code": mask_geo_code, "mask_ground_truth": mask_ground_truth}
        else:
            pos_id = torch.LongTensor([pos_id])
            pos_geo_code = torch.LongTensor([hash2index[hash_] for hash_ in start_token + pos_geo_code])
            data = {"query_text": query, "pos_geo_code": pos_geo_code, "pos_id": pos_id}
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
    def __init__(self, pretrain_dim, model_dim):
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


        self.token_encoder = nn.Sequential(
            nn.LayerNorm(pretrain_dim),
            nn.Linear(pretrain_dim, pretrain_dim),
            nn.LeakyReLU(),
            nn.Linear(pretrain_dim, self.dim),
        )
        token_moe = MoE(
            dim = self.dim,
            num_experts = 8,               # increase the experts (# parameters) of your model without increasing computation
            gating_top_n = 2,               # default to top 2 gating, but can also be more (3 was tested in the paper with a lower threshold)
            threshold_train = 0.2,          # at what threshold to accept a token to be routed to second expert and beyond - 0.2 was optimal for 2 expert routing, and apparently should be lower for 3
            threshold_eval = 0.2,
            capacity_factor_train = 1.25,   # experts have fixed capacity per batch. we need some extra capacity in case gating is not perfectly balanced.
            capacity_factor_eval = 2.,      # capacity_factor_* should be set to a value >=1
            balance_loss_coef = 1e-2,       # multiplier on the auxiliary expert balancing auxiliary loss
            router_z_loss_coef = 1e-3,      # loss weight for router z-loss
        )
        self.token_moe_block = SparseMoEBlock(
            token_moe,
            add_ff_before = True,
            add_ff_after = True
        )


        self.geo_encoder = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.dim),
            nn.LeakyReLU(),
            nn.Linear(self.dim, self.dim),
        )
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.dim, nhead=4, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)

        self.geohash_embedding = nn.Embedding(num_embeddings=self.geo_char_len, embedding_dim=self.dim)

        moe = MoE(
            dim = self.dim,
            num_experts = 8,               # increase the experts (# parameters) of your model without increasing computation
            gating_top_n = 2,               # default to top 2 gating, but can also be more (3 was tested in the paper with a lower threshold)
            threshold_train = 0.2,          # at what threshold to accept a token to be routed to second expert and beyond - 0.2 was optimal for 2 expert routing, and apparently should be lower for 3
            threshold_eval = 0.2,
            capacity_factor_train = 1.25,   # experts have fixed capacity per batch. we need some extra capacity in case gating is not perfectly balanced.
            capacity_factor_eval = 2.,      # capacity_factor_* should be set to a value >=1
            balance_loss_coef = 1e-2,       # multiplier on the auxiliary expert balancing auxiliary loss
            router_z_loss_coef = 1e-3,      # loss weight for router z-loss
        )
        self.moe_block = SparseMoEBlock(
            moe,
            add_ff_before = True,
            add_ff_after = True
        )

        

    def get_geo_embedding(self, token_embedding, token_mask, geo_code, mode):
        geo_embedding = self.geohash_embedding(geo_code)
        cat_embedding = torch.cat([token_embedding, geo_embedding], dim=1)
        cat_embedding = self.pos_encoder(cat_embedding * math.sqrt(self.dim))
        mask = torch.cat([token_mask, torch.ones((geo_embedding.size(0), geo_embedding.size(1))).to(token_mask.device)], dim=1)
        mask = ~mask.bool()
        cat_embedding = self.transformer_encoder(cat_embedding, src_key_padding_mask=mask)
        cat_embedding, aux_loss, _, _ = self.moe_block(cat_embedding)
        out = cat_embedding[:, -geo_embedding.size(1):, :]
        return out, aux_loss
    
    def get_text_embedding(self, text_embedding):
        text_embedding = self.text_encoder(text_embedding)
        return text_embedding

    def get_token_embedding(self, token_embedding):
        out, aux_loss, _, _ = self.token_moe_block(token_embedding)
        return out, aux_loss

    def forward(self, text_embedding, geo_code, mode):
        text_embedding = self.text_encoder(text_embedding)
        geo_embedding = self.get_geo_embedding(geo_code, mode)
        
        return {"text_embedding": text_embedding, "geo_embedding": geo_embedding}

class ULPModel(nn.Module):
    def __init__(self, embedding_model, embedding_model_name, device):
        super().__init__()
        self.embedding_model_name = embedding_model_name
        self.pretrain_nlp_model = embedding_model
        for param in self.pretrain_nlp_model.parameters():
            param.requires_grad = False
        
        self.pretrain_dim = list(self.pretrain_nlp_model.state_dict().items())[-1][-1].size(0)
        self.dim = 1024
        self.encoder = Encoder(self.pretrain_dim, self.dim)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=self.dim, nhead=4, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=4)

        self.geo_char_len = len(geo_code_char)
        self.vocab_size = self.geo_char_len - 2
        self.predict_layer = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.dim * 2),
            nn.LeakyReLU(),
            torch.nn.Linear(self.dim * 2, self.vocab_size)
        )

        self.device = device

        self.mask_geo_predict_layer = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.dim),
            nn.LeakyReLU(),
            nn.Linear(self.dim, self.vocab_size),
        )

    def get_pretrain_embedding(self, text_data):
        with torch.no_grad():
            if "gte" in self.embedding_model_name:
                pretrain_embedding = self.pretrain_nlp_model(**text_data).last_hidden_state
                pretrain_embedding = self.average_pool(pretrain_embedding, text_data['attention_mask'])
            elif "bge" in self.embedding_model_name:
                pretrain_result = self.pretrain_nlp_model(**text_data)[0]
                token_embedding = pretrain_result[:, 1:]
                pretrain_embedding = pretrain_result[:, 0]
                pretrain_embedding = torch.nn.functional.normalize(pretrain_embedding, p=2, dim=-1)
            elif "bert" in self.embedding_model_name:
                pretrain_embedding = self.pretrain_nlp_model(**text_data).last_hidden_state
                pretrain_embedding = pretrain_embedding[:, 0, :].squeeze()
            elif "MiniCPM" in self.embedding_model_name:
                outputs = self.pretrain_nlp_model(**text_data)
                attention_mask = text_data["attention_mask"]
                s = torch.sum(outputs.last_hidden_state * attention_mask.unsqueeze(-1).float(), dim=1)
                d = attention_mask.sum(dim=1, keepdim=True).float()
                reps = s / d
                pretrain_embedding = F.normalize(reps, p=2, dim=1)
            else:
                raise Exception("Unknow Pretrain Model")
        return {"pretrain_embedding": pretrain_embedding, "token_embedding": token_embedding}
    
    def beam_search(self, max_length, memory):
        beam_width = 5
        batch_size = memory.size(0)
        beam_scores = torch.zeros((batch_size, beam_width), device=self.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view(-1)
        generated_hyps = [BeamHypotheses(num_beams=beam_width, max_length=max_length, length_penalty=0.7, early_stopping=False) for _ in range(batch_size)]
        input_ids = torch.full((batch_size * beam_width, 1), sos_token_id, dtype=torch.long, device=self.device)
        for i in range(max_length):
            input_embeddings = self.encoder.geohash_embedding(input_ids).view(batch_size, beam_width, i+1, -1)
            decoder_mask = self._generate_square_subsequent_mask(i+1)
            outputs_list = []
            for j in range(beam_width):
                input_embeddings_ = self.encoder.pos_encoder(input_embeddings[:,j,...] * math.sqrt(self.dim))
                outputs = self.transformer_decoder(input_embeddings_, memory, tgt_mask=decoder_mask)
                outputs = self.predict_layer(outputs)
                outputs_list.append(outputs[:, -1:, :])
            next_token_logits = torch.cat(outputs_list, dim=1).view(batch_size * beam_width, -1)
            scores = F.log_softmax(next_token_logits, dim=-1)
            next_scores = scores + beam_scores[:, None].expand_as(scores)
            next_scores = next_scores.view(batch_size, beam_width * self.vocab_size)
            next_scores, next_tokens = torch.topk(next_scores, beam_width, dim=-1, largest=True, sorted=True)
            next_batch_beam = []
            for batch_idx in range(batch_size):
                next_sent_beam = [] 
                for beam_token_rank, (beam_token_id, beam_token_score) in enumerate(zip(next_tokens[batch_idx], next_scores[batch_idx])):
                    beam_id = beam_token_id // self.vocab_size # 1
                    token_id = beam_token_id % self.vocab_size # 1
                    effective_beam_id = batch_idx * beam_width + beam_id

                    if i == max_length-1:
                        is_beam_token_worse_than_top_num_beams = beam_token_rank >= beam_width
                        if is_beam_token_worse_than_top_num_beams:
                            continue
                        generated_hyps[batch_idx].add(input_ids[effective_beam_id].clone(), beam_token_score.item())
                    else:
                        next_sent_beam.append((beam_token_score, token_id, effective_beam_id))
                    
                    if len(next_sent_beam) == beam_width:
                        break

                next_batch_beam.extend(next_sent_beam)

            beam_scores = beam_scores.new([x[0] for x in next_batch_beam]) # (num_beams * batch_size)
            beam_tokens = input_ids.new([x[1] for x in next_batch_beam]) # (num_beams * batch_size)
            beam_idx = input_ids.new([x[2] for x in next_batch_beam]) # (num_beams * batch_size)

            input_ids = input_ids[beam_idx, :] # (num_beams * batch_size, seq_len)
            input_ids = torch.cat([input_ids, beam_tokens.unsqueeze(1)], dim=-1)

        output_num_return_sequences_per_batch = 1
        output_batch_size = output_num_return_sequences_per_batch * batch_size
        sent_lengths = input_ids.new(output_batch_size)
        best = []
        for i, hypotheses in enumerate(generated_hyps):
            sorted_hyps = sorted(hypotheses.beams, key=lambda x: x[0])
            for j in range(output_num_return_sequences_per_batch):
                effective_batch_idx = output_num_return_sequences_per_batch * i + j
                best_hyp = sorted_hyps.pop()[1]
                sent_lengths[effective_batch_idx] = len(best_hyp)
                best.append(best_hyp)
        outputs = torch.stack(best).view(batch_size, -1, max_length).long()
        return outputs
    
    def _generate_square_subsequent_mask(self, sz):
        return torch.log(torch.tril(torch.ones(sz,sz)))
    
    def forward(self, data, stage):
        if stage == "train":
            query_text, pos_geo_code, mask_geo_code = data["query_text"], data["pos_geo_code"], data["mask_geo_code"]
        elif stage == "test":
            query_text, pos_geo_code = data["query_text"], data["pos_geo_code"]
        pretrain_result = self.get_pretrain_embedding(query_text)
        query_embedding, token_embedding = pretrain_result["pretrain_embedding"], pretrain_result["token_embedding"]
        query_token_embedding, token_aux_loss = self.encoder.get_token_embedding(token_embedding)

        if stage == "train":
            token_mask = data["query_text"]["attention_mask"][:, 1:]
            mask_geo_embedding, aux_loss = self.encoder.get_geo_embedding(query_token_embedding, token_mask, mask_geo_code, "mask")
            mask_geo_output = self.mask_geo_predict_layer(mask_geo_embedding)
            
            decoder_geo_embedding = self.encoder.geohash_embedding(pos_geo_code[:,:-1])
            decoder_mask = self._generate_square_subsequent_mask(decoder_geo_embedding.size(1))
            decoder_output = self.transformer_decoder(decoder_geo_embedding, query_token_embedding, tgt_mask=decoder_mask)
            geo_output = self.predict_layer(decoder_output)
            
            return {"geo_output": geo_output, "mask_geo_output": mask_geo_output, "aux_loss": aux_loss, "token_aux_loss": token_aux_loss}
        elif stage == "test":
            max_length = data["pos_geo_code"].size(1)
            geo_code = self.beam_search(max_length, query_token_embedding)
            return {"geo_code": geo_code}
    
class OursLossFunction(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss(ignore_index=len(geo_code_char))
        self.mask_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

        self.sigma = torch.nn.Parameter(torch.ones(4, device=device))
        self.infonce = InfoNCE()
        self.a_value = 0.9
        self.padding = 0
        self.continuous_rewards = [self.padding] + (1 / torch.tensor([self.integral(0, i+1) for i in range(len(geo_code_char))])).tolist()
        weight = torch.linspace(0, 11, 12, device=device)
        self.correct_weight = torch.pow(self.a_value, weight)
        self.error_weight = self.correct_weight + 1
        self.device = device
    
    def integral(self, a, b, num_points=2000):
        x = torch.linspace(a, b, num_points)
        dx = (b - a) / num_points
        y = torch.pow(self.a_value, x)
        return torch.sum(y * dx)


    def forward(self, outputs, data):
        mask_loss = self.mask_loss_fn(outputs["mask_geo_output"].transpose(1, 2), data["mask_ground_truth"])
        targets = data["pos_geo_code"][:,1:]
        geo_output = outputs["geo_output"]
        cross_loss = self.cross_entropy_loss(geo_output.transpose(1, 2), targets)
        geo_loss = cross_loss
        weighted_loss = 0.5 * torch.stack([geo_loss, mask_loss, outputs["aux_loss"], outputs["token_aux_loss"]]) / self.sigma ** 2
        weighted_loss = weighted_loss.sum() + torch.log(self.sigma.prod())
        
        return {"geo_loss": weighted_loss, "mask_loss": mask_loss}

class Trainner:
    def __init__(self, embedding_model_name, tokenizer):
        self.device = torch.device(f"cuda:{device_idx}" if torch.cuda.is_available() else "cpu")
        self.tokenizer = tokenizer
        self.embedding_model = AutoModel.from_pretrained(embedding_model_name, cache_dir="../../../huggingface/hub/")
        self.model = ULPModel(self.embedding_model, embedding_model_name, device=self.device).to(self.device)
        self.our_loss_fn = OursLossFunction(self.device)
        self.optimizer = torch.optim.AdamW(list(self.model.parameters()) + list(self.our_loss_fn.parameters()), lr=5e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(optimizer=self.optimizer, step_size=4, gamma=0.9)
        self.scaler = torch.amp.GradScaler()
        self.save_dir = "./save/checkpoints"
        self.max_keep = 5
        self.K_list = [1, 3, 5, 10]
        self.correct_dict = {K: 0 for K in self.K_list}
        self.test_vec_db = None
        self.gpu_index = None
        self.docid2index = {doc_id: index for index, doc_id in enumerate(poi_df["doc_id"])}
        
        self.train_dataset = ULPDataset(poi_df=poi_df, data_df=train_df, mode="train")
        self.train_loader = DataLoader(self.train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
        
        self.val_dataset = ULPDataset(poi_df=poi_df, data_df=val_df, mode="val")
        self.val_loader = DataLoader(self.val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
        
        self.test_dataset = ULPDataset(poi_df=poi_df, data_df=test_df, mode="test")
        self.test_loader = DataLoader(self.test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
        
    def train(self, epoch):
        self.model.train()
        result = self.iteration(epoch, self.train_loader, mode="train")
        self.scheduler.step()
        return result
    
    def val(self, epoch):
        self.model.eval()
        result = self.iteration(epoch, self.val_loader, mode="val")
        return result
    
    def test(self, epoch):
        self.model.eval()
        result = self.iteration(epoch, self.test_loader, mode="test")
        return result
    
    def trans_data(self, data, mode):
        for key, value in data.items():
            if "text" in key or "address" in key:
                if key == "neg_address":
                    if mode == "test": continue
                    value = self.tokenizer(list(chain(*value)), padding=True, truncation=True, return_tensors='pt')
                else:
                    value = self.tokenizer(value, padding=True, truncation=True, return_tensors='pt')
            data[key] = value.to(self.device)
        return data
    
    def generate_geo(self):
        result = dict()
        for i, data in enumerate(tqdm(self.test_loader, desc=f"generate")):
            data = self.trans_data(data, "test")
            with torch.no_grad():
                outputs = self.model(data, stage="test")
            geo_code = outputs["geo_code"].squeeze(1)[:,1:].tolist()
            geo_code = ["".join([index2hash[c] for c in code]) for code in geo_code]

            for index, code in zip(data["index"].tolist(), geo_code):
                result[index] = {"geo_code": code}
        np.save(f"./save/geo_memory_{dataset_name}.npy", arr=result)
    
    def metrics(self, outputs, data):
        geo_code = outputs["geo_code"]
        geo_outputs = [geohash.decode("".join(row)) for row in index2hash_func(outputs["geo_code"][:, 0, 1:].tolist())]
        pos_geo = poi_df.iloc[data["pos_id"].squeeze(1).tolist()][["lat", "lng"]].values.tolist()
        distance_list = []
        for geo, pos in zip(geo_outputs, pos_geo):
            distance = haversine(geo, pos, unit=Unit.KILOMETERS)
            distance_list.append(distance)
        ground_truth = data["pos_geo_code"]
        equal_matrix = (geo_code.squeeze(1) == ground_truth)
        correct_count_dict = dict()
        for i in range(1, equal_matrix.size(-1)+1):
            correct_count = (equal_matrix[:, :i].sum(-1) == i).sum().item()
            correct_count_dict[i] = correct_count
        return {"correct_count_dict": correct_count_dict, "distance_list": distance_list}
        
    def iteration(self, epoch, data_loader, mode):
        epoch_loss_list = []
        prefix_loss_list = []
        correct_loss_list = []
        error_loss_list = []
        mask_loss_list = []
        total = 0
        correct_dict = {i: 0 for i in range(1, 14)}
        distance_list_result = []
        for index, data in enumerate(tqdm(data_loader, desc=f"{mode}")):
            data = self.trans_data(data, mode)
            if mode == "train":
                outputs = self.model(data, stage="train")
                loss_dict = self.our_loss_fn(outputs, data)
                geo_loss = loss_dict["geo_loss"]
                mask_loss = loss_dict["mask_loss"]
                geo_loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                epoch_loss_list.append(geo_loss.item())
                mask_loss_list.append(mask_loss.item())

            if mode == "test" or mode == "val":
                with torch.no_grad():
                    outputs = self.model(data, stage="test")
                geo_code = outputs["geo_code"]
                total += geo_code.size(0)
                metric_results = self.metrics(outputs, data)
                correct_count_dict = metric_results["correct_count_dict"]
                distance_list = metric_results["distance_list"]
                distance_list_result.extend(distance_list)
                for prefix_len, count in correct_count_dict.items():
                    correct_dict[prefix_len] += count
        if mode == "train":
            logging.info(f"{epoch} | total: {np.mean(epoch_loss_list)} | mask: {np.mean(mask_loss_list)}")
            return {"geo_loss": np.mean(epoch_loss_list)}
        elif mode == "test" or mode == "val":
            correct_ratio_dict = {prefix_len: f"{(count / total * 100):.2f}" for prefix_len, count in correct_dict.items()}
            print(correct_ratio_dict)
            logging.info(correct_ratio_dict)
            distance_tensor = torch.Tensor(distance_list_result)
            acc1 = ((distance_tensor < 1).sum() / len(distance_tensor) * 100).item()
            acc5 = ((distance_tensor < 5).sum() / len(distance_tensor) * 100).item()
            acc10 = ((distance_tensor < 10).sum() / len(distance_tensor) * 100).item()
            logging.info(f"{mode} | {epoch} | mean: {np.mean(distance_list_result)} | median: {np.median(distance_list_result)} | std: {np.std(distance_list_result)} | acc1: {acc1} | acc5: {acc5} | acc10: {acc10}")
            print(f"{mode} | {epoch} | mean: {np.mean(distance_list_result)} | median: {np.median(distance_list_result)} | std: {np.std(distance_list_result)} | acc1: {acc1} | acc5: {acc5} | acc10: {acc10}")
            return {"mean": np.mean(distance_list_result)}
    
    def start(self):
        # self.generate_geo()
        best_mean_distance = np.inf
        epochs = 200

        for epoch in range(epochs):
            train_result = self.train(epoch)  # train
            val_result = self.val(epoch)
            if val_result["mean"] < best_mean_distance:
                test_result = self.test(epoch)
                best_mean_distance = val_result["mean"]
            
                model_state_dict = dict()
                for key, value in self.model.state_dict().items():
                    if "pretrain_nlp_model" in key: continue
                    model_state_dict[key] = value

                checkpoint = {
                    "epoch": epoch, 
                    "model": model_state_dict,
                    "optimizer": self.optimizer.state_dict(),
                    "scheduler": self.scheduler.state_dict(),
                    "loss_fn": self.our_loss_fn.state_dict(),
                }

                torch.save(checkpoint, f"./save/checkpoint_v{v}/checkpoint_{epoch}.pth")

def dt_converter(*args):
    now = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=8)
    return now.timetuple()

if __name__ == '__main__':
    dataset_name = "geoglue"
    device_idx = 0
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

    v = 24
    save_dir = f"./save/checkpoint_v{v}"
    if not os.path.exists(save_dir): os.makedirs(save_dir)

    logging_format = "%(asctime)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    logging.Formatter.converter = dt_converter
    logging.basicConfig(filename=f"./save/pretrain_{device_idx}_v{v}.log", level=logging.INFO, format=logging_format, datefmt=date_format)

    BATCH_SIZE = 200
    NUM_WORKERS = 8
    MAX_LEN = 100
    NEG_COUNT = 20

    poi_df = pd.read_csv(f"../recall_data/{dataset_name}/dataset/poi.csv", header=0)
    poi_df["address"] = poi_df["address"].str.slice(0, MAX_LEN)
    train_df = pd.read_csv(f"../recall_data/{dataset_name}/dataset/train.csv", header=0)
    train_df["query"] = train_df["query"].str.slice(0, MAX_LEN)
    val_df = pd.read_csv(f"../recall_data/{dataset_name}/dataset/eval.csv", header=0)
    val_df["query"] = val_df["query"].str.slice(0, MAX_LEN)
    test_df = pd.read_csv(f"../recall_data/{dataset_name}/dataset/test.csv", header=0)
    test_df["query"] = test_df["query"].str.slice(0, MAX_LEN)
    
    index2hash_func = np.vectorize(index2hash.get)

    embedding_model_name = 'BAAI/bge-m3'
    tokenizer = AutoTokenizer.from_pretrained(embedding_model_name, cache_dir="../../../huggingface/hub/")
    trainner = Trainner(embedding_model_name, tokenizer)
    trainner.start()