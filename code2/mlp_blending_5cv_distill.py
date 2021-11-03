import os
import jieba
import time
import random
import scipy
import json
import pickle
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics.pairwise import cosine_similarity
from zipfile import ZIP_DEFLATED, ZipFile
from multiprocessing import Pool

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn.modules.loss import _WeightedLoss
import torch.nn.functional as F

from utils import LoadDataset, AverageMeter, EarlyStopping

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)

DEBUG = False

DATA_PATH = '../data/new_data/'
SAVE_PATH = '../final_output/mlp_blending_5cv_distill1015'
if not os.path.exists(SAVE_PATH):
    os.mkdir(SAVE_PATH)
if DEBUG:
    SAVE_PATH = f'{SAVE_PATH}/debug'
    if not os.path.exists(SAVE_PATH):
        os.mkdir(SAVE_PATH)

BATCH_SIZE = 1024
EARLYSTOP_NUM = 1
EPOCHS = 10
LR = 1e-3
NFOLDS = 5
if DEBUG:
    EPOCHS = 1
    NFOLDS = 2

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(seed=42)
dataset = LoadDataset(data_path=DATA_PATH, debug=DEBUG)

file_list = [
    '../final_output/light_lxmert1001/',
    '../final_output/gru_lxmert1001/',
    '../final_output/lxmert1001/',
    '../final_output/enhancedrcnn1001/',

    '../final_output/lstm_nextvlad_siamese1001/',
    # '../final_output/lstm_trm_siamese1001/',
    # '../final_output/lstm_inception_siamese1001/',
    # '../final_output/bert_nextvlad_siamese1001/',

    # '../final_output/lstm_nextvlad_denseSE1001/',
    # '../final_output/lstm_trm_denseSE1001/',
    # '../final_output/lstm_inception_denseSE1001/',

    '../final_output/lstm1001/',

    '../final_output/inception_lxmert1001/',
    '../final_output/esim1011_5/',

    # '../final_output/light_cross_lxmert1013/',
    # '../final_output/gru_vilbert1013/',
]

pairwise_emb, test_emb = [], []
for path in tqdm(file_list, desc='Load Embeddings'):
    if DEBUG:
        pairwise_emb_slice = np.load(f'{path}/debug/distill_pairwise_pred.npy')
        test_emb_slice = np.load(f'{path}/debug/distill_test_pred.npy')
    else:
        pairwise_emb_slice = np.load(f'{path}/distill_pairwise_pred.npy')
        test_emb_slice = np.load(f'{path}/distill_test_pred.npy')
    # print(pairwise_emb_slice.shape), print(test_emb_slice.shape)
    pairwise_emb.append(pairwise_emb_slice)
    test_emb.append(test_emb_slice)
pairwise_emb = np.concatenate(pairwise_emb, axis=1).reshape((len(dataset.pairwise_ids), 256, -1))
test_emb = np.concatenate(test_emb, axis=1).reshape((len(dataset.test_ids), 256, -1))
print('pairwise_emb: ', pairwise_emb.shape)
print('test_emb: ', test_emb.shape)

# for i in range(pairwise_emb.shape[-1]-1):
#     for j in range(i+1, pairwise_emb.shape[-1]):
#         sim_list = []
#         for k in range(pairwise_emb.shape[0]):
#             sim = cosine_similarity([pairwise_emb[k, :, i]], [pairwise_emb[k, :, j]])[0][0]
#             sim_list.append(sim)
#         sim_score = np.mean(sim_list)
#         i_model = file_list[i].split('/')[-2]
#         j_model = file_list[j].split('/')[-2]
#         print(i_model, j_model, sim_score)

id2feat = {}
for id, emb in zip(dataset.pairwise_ids, pairwise_emb):
    id2feat[id] = emb
for id, emb in zip(dataset.test_ids, test_emb):
    id2feat[id] = emb

class PairDataset:
    def __init__(self, df, id2feat):
        self.id1 = df['id1'].values
        self.id2 = df['id2'].values
        self.label = df['label'].values

        self.id2feat = id2feat

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        feat1 = self.id2feat[self.id1[idx]]
        feat2 = self.id2feat[self.id2[idx]]

        return {
            'feat1': torch.tensor(feat1, dtype=torch.float),
            'feat2': torch.tensor(feat2, dtype=torch.float),
            'label': torch.tensor(self.label[idx], dtype=torch.float),
        }

class EmbDataset:
    def __init__(self, id_list, id2feat):
        self.ids = id_list
        self.id2feat = id2feat

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        feat = self.id2feat[self.ids[idx]]

        return {
            'feat': torch.tensor(feat, dtype=torch.float),
        }

class PairSiameseModel(nn.Module):
    def __init__(self, feat_dim):
        super(PairSiameseModel, self).__init__()
        dense_dim = 64
        self.dense1 = nn.Linear(feat_dim, dense_dim, bias=False) # , bias=False
        self.dense2 = nn.Linear(dense_dim, dense_dim, bias=False)
        self.dense3 = nn.Linear(dense_dim, 1, bias=False)
        self.dropout = nn.Dropout(0.2)

    def encoder(self, feat):
        # encoded_output = self.dropout(feat)
        # encoded_output = torch.relu(encoded_output)
        encoded_output = self.dense1(feat)
        # encoded_output = self.dropout(encoded_output)
        encoded_output = self.dense2(encoded_output)
        # encoded_output = self.dropout(encoded_output)
        encoded_output = self.dense3(encoded_output).squeeze(-1)
        encoded_output = torch.nn.functional.normalize(encoded_output, p=2, dim=1)
        # print('encoded_output: ', encoded_output.shape)
        return encoded_output

    def forward(self, feat1, feat2):
        encoded_emb1 = self.encoder(feat1)
        encoded_emb2 = self.encoder(feat2)
        return torch.cosine_similarity(encoded_emb1, encoded_emb2)

def pairwise_train_fn(data_loader, model, optimizer, loss_fn, device, scheduler=None, debug=False):
    model.train()
    losses = AverageMeter()

    tk0 = tqdm(data_loader, total=len(data_loader), desc="Train")
    for bi, d in enumerate(tk0):
        torch.cuda.empty_cache()
        feat1 = d["feat1"].to(device, dtype=torch.float)
        feat2 = d["feat2"].to(device, dtype=torch.float)
        labels = d["label"].to(device, dtype=torch.float)

        model.zero_grad()
        outputs = model(feat1, feat2)
        # print('outputs: ', outputs.shape)
        # print('labels: ', outputs.shape)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()
        torch.cuda.empty_cache()

        losses.update(loss.item(), labels.size(0))
        tk0.set_postfix(loss=losses.avg)
        if debug:
            if bi >= 10:
                break

def pairwise_infer_fn(data_loader, model, device):
    model.eval()

    pred_list = []
    # ema.apply_shadow()
    with torch.no_grad():
        tk0 = tqdm(data_loader, total=len(data_loader), desc="Pred")
        for bi, d in enumerate(tk0):
            torch.cuda.empty_cache()
            feat1 = d["feat1"].to(device, dtype=torch.float)
            feat2 = d["feat2"].to(device, dtype=torch.float)

            outputs = model(feat1, feat2)
            # outputs = torch.sigmoid(outputs).cpu().detach().numpy().astype(float)
            outputs = outputs.cpu().detach().numpy().astype(float)
            pred_list.append(outputs)

    return np.concatenate(pred_list).reshape(-1)

def pairwise_emb_infer_fn(data_loader, model, encode_dim, device):
    model.eval()

    pred_list = []
    # ema.apply_shadow()
    with torch.no_grad():
        tk0 = tqdm(data_loader, total=len(data_loader), desc="Pred")
        for bi, d in enumerate(tk0):
            torch.cuda.empty_cache()
            feat = d["feat"].to(device, dtype=torch.float)

            outputs = model.module.encoder(feat)
            # outputs = torch.sigmoid(outputs).cpu().detach().numpy().astype(float)
            outputs = outputs.cpu().detach().numpy().astype(float)
            pred_list.append(outputs)

    return np.concatenate(pred_list).reshape(-1, encode_dim)

def pairwise_loss_fn(output, label):
    loss_fct = nn.MSELoss()
    # loss_fct = nn.L1Loss()
    # loss_fct = nn.SmoothL1Loss()
    # loss_fct = nn.BCEWithLogitsLoss()
    # loss_fct = nn.BCELoss()
    # loss_fct = FocalBCELoss()
    # loss_fct = SpearmanLoss()
    return loss_fct(output, label)

oof_pred = np.zeros(len(dataset.label_df))
pairwise_pred = np.zeros((len(dataset.pairwise_ids), 256))
test_pred = np.zeros((len(dataset.test_ids), 256))

pairwise_set = EmbDataset(id_list=dataset.pairwise_ids, id2feat=id2feat)
pairwise_loader = DataLoader(pairwise_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
test_set = EmbDataset(id_list=dataset.test_ids, id2feat=id2feat)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# kf = KFold(n_splits=NFOLDS, shuffle=True, random_state=42)
kf = GroupKFold(n_splits=NFOLDS)
# for _fold, (train_idx, valid_idx) in enumerate(kf.split(dataset.label_df, dataset.label_df)):
for _fold, (train_idx, valid_idx) in enumerate(kf.split(dataset.label_df, dataset.label_df, groups=dataset.label_df.id1.values)):
    seed_everything(seed=42 + _fold)
    print(f'\nFold{_fold}: {len(train_idx)}|{len(valid_idx)}')
    train_set = PairDataset(dataset.label_df.loc[train_idx], id2feat)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    valid_set = PairDataset(dataset.label_df.loc[valid_idx], id2feat)
    valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    # for batch in train_loader:
    #     print(batch)
    #     break

    torch.cuda.empty_cache()
    device = torch.device("cuda:0")
    model = PairSiameseModel(feat_dim=pairwise_emb.shape[-1])
    model.to(device)
    model = nn.DataParallel(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = None

    best_score = 0
    MODEL_WEIGHT = f"{SAVE_PATH}/blend_model{_fold}.bin"
    es = EarlyStopping(patience=EARLYSTOP_NUM, mode="max")
    for epoch in range(EPOCHS):
        pairwise_train_fn(train_loader, model, optimizer, pairwise_loss_fn, device, scheduler=scheduler, debug=False)
        valid_pred = pairwise_infer_fn(valid_loader, model, device)
        valid_spearmanr = scipy.stats.spearmanr(valid_pred, dataset.label_df.loc[valid_idx].label.values).correlation

        print(f'Epoch {epoch + 1}/{EPOCHS} valid_spearmanr={valid_spearmanr:.4f}')

        if best_score < valid_spearmanr:
            best_score = valid_spearmanr
        es(valid_spearmanr, model, model_path=MODEL_WEIGHT)
        if es.early_stop:
            print("Early stopping")
            break

    model.load_state_dict(torch.load(MODEL_WEIGHT))

    oof_pred[valid_idx] = pairwise_infer_fn(valid_loader, model, device)

    pairwise_pred += pairwise_emb_infer_fn(pairwise_loader, model, 256, device) / NFOLDS
    test_pred += pairwise_emb_infer_fn(test_loader, model, 256, device) / NFOLDS

oof_spearmanr = scipy.stats.spearmanr(oof_pred, dataset.label_df.label.values).correlation
print(f'oof_spearmanr={oof_spearmanr:.4f}')

emb_pred = {}
for vid, emb in zip(dataset.test_ids, test_pred):
    emb_pred[str(vid)] = emb.tolist()
with open('result.json', 'w') as f:
    json.dump(emb_pred, f)
with ZipFile(f'{SAVE_PATH}/mlp{NFOLDS}cv_distill_blend{oof_spearmanr:.4f}.zip', 'w', compression=ZIP_DEFLATED) as zip_file:
    zip_file.write('result.json')

# oof_spearmanr=0.9386