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
from sklearn.model_selection import KFold
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

from transformers import BertConfig, BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup

from modules import SoftmaxAttention, RNNDropout, Seq2SeqEncoder
from utils import jieba_seg, parallelize_df_func, save_pickle, load_pickle, save_json, load_json,\
    seed_everything, AverageMeter, EarlyStopping, LoadDataset, FocalBCELoss, caculate_spearmanr_score

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)

DEBUG = False
POINTWISE_TRAIN = True
PAIRWISE_TRAIN = True
DISTILL_TRAIN = True
DISTILL_TRAIN2 = False

W2V_DIM = 300
HEAD_NUM = 4
LAYER_NUM = 1
ENCODE_DIM = 256

DATA_PATH = '../data/new_data/'
W2V_PATH = '../data/new_data/title_w2v_final'
# ROOT_PRETRAINED_PATH = '/search/odin/lida/pytorch_pretrained_models_nlp'
# PRETRAINED_MODEL_PATH = f'{ROOT_PRETRAINED_PATH}/bert-base-chinese/'
SAVE_PATH = '../final_output/esim1011_5'
if not os.path.exists(SAVE_PATH):
    os.mkdir(SAVE_PATH)
if DEBUG:
    SAVE_PATH = f'{SAVE_PATH}/debug'
    if not os.path.exists(SAVE_PATH):
        os.mkdir(SAVE_PATH)

########################################################################################################################
##### Data fn
class PointDataset:
    def __init__(self, id_list, dataset, id2seq, max_len, is_test=False):
        self.ids = id_list
        self.id2title = dataset.id2title
        self.id2asr = dataset.id2asr
        self.id2frame_num = dataset.id2frame_num
        self.id2tag = dataset.id2tag
        self.id2cate = dataset.id2cate
        self.tag2id = dataset.tag2id
        self.id2seq = id2seq

        self.max_len = max_len
        self.video_path = f'{dataset.data_path}/video'

        self.is_test = is_test

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        this_id = self.ids[idx]
        title = self.id2title[this_id]
        asr = self.id2asr[this_id]
        video = np.load(f'{self.video_path}/{this_id}.npy')
        frame_num = self.id2frame_num[this_id]
        if frame_num < 32:
            video[frame_num:] = 0.

        title_seq = self.id2seq[this_id]
        title_length = max(1, min(len(title_seq), self.max_len))
        if len(title_seq) >= self.max_len:
            title_seq = title_seq[:self.max_len]
        else:
            title_seq += [0] * (self.max_len - len(title_seq))

        if not self.is_test:
            label = self.tag2multihot(this_id)
            # print('label: ', label.shape, np.sum(label))

        sample = {
            'id': this_id,
            'title_seq': torch.tensor(title_seq, dtype=torch.long),
            'title_length': torch.tensor(title_length, dtype=torch.long),
            'video': torch.tensor(video, dtype=torch.float),
            'video_length': torch.tensor(frame_num, dtype=torch.long),
        }
        if not self.is_test:
            sample['label'] = torch.tensor(label, dtype=torch.float)
        return sample

    def tag2multihot(self, _id):
        tag_list = str(self.id2tag[_id]).split(' ')
        res = np.zeros(len(self.tag2id))
        for tag in tag_list:
            if tag in self.tag2id:
                # print('***', self.tag2id[tag])
                res[self.tag2id[tag]] = 1
        return res

class PairDataset:
    def __init__(self, df, dataset, id2seq, max_len):
        self.id1 = df['id1'].values
        self.id2 = df['id2'].values
        self.label = df['label'].values

        self.dataset = dataset
        self.id2frame_num = dataset.id2frame_num
        self.id2seq = id2seq
        self.max_len = max_len
        self.video_path = f'{dataset.data_path}/video'

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        title_seq1 = self.id2seq[str(self.id1[idx])]
        title_length1 = max(1, min(len(title_seq1), self.max_len))
        if len(title_seq1) >= self.max_len:
            title_seq1 = title_seq1[:self.max_len]
        else:
            title_seq1 += [0] * (self.max_len - len(title_seq1))

        video_matrix1 = np.load(f'{self.video_path}/{self.id1[idx]}.npy')
        frame_num1 = self.id2frame_num[self.id1[idx]]
        if frame_num1 < 32:
            video_matrix1[frame_num1:] = 0.

        title_seq2 = self.id2seq[str(self.id2[idx])]
        title_length2 = max(1, min(len(title_seq2), self.max_len))
        if len(title_seq2) >= self.max_len:
            title_seq2 = title_seq2[:self.max_len]
        else:
            title_seq2 += [0] * (self.max_len - len(title_seq2))

        video_matrix2 = np.load(f'{self.video_path}/{self.id2[idx]}.npy')
        frame_num2 = self.id2frame_num[self.id2[idx]]
        if frame_num2 < 32:
            video_matrix2[frame_num2:] = 0.

        return {
            'title_seq1': torch.tensor(title_seq1, dtype=torch.long),
            'title_length1': torch.tensor(title_length1, dtype=torch.long),
            'video_matrix1': torch.tensor(video_matrix1, dtype=torch.float),
            'video_length1': torch.tensor(frame_num1, dtype=torch.long),
            'title_seq2': torch.tensor(title_seq2, dtype=torch.long),
            'title_length2': torch.tensor(title_length2, dtype=torch.long),
            'video_matrix2': torch.tensor(video_matrix2, dtype=torch.float),
            'video_length2': torch.tensor(frame_num2, dtype=torch.long),
            'label': torch.tensor(self.label[idx], dtype=torch.float),
        }

class EmbDataset:
    def __init__(self, id_list, dataset, id2seq, max_len):
        self.ids = id_list
        self.dataset = dataset
        self.id2frame_num = dataset.id2frame_num
        self.id2seq = id2seq
        self.max_len = max_len
        self.video_path = f'{dataset.data_path}/video'

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        title_seq = self.id2seq[str(self.ids[idx])]
        title_length = max(1, min(len(title_seq), self.max_len))
        if len(title_seq) >= self.max_len:
            title_seq = title_seq[:self.max_len]
        else:
            title_seq += [0] * (self.max_len - len(title_seq))

        video_matrix = np.load(f'{self.video_path}/{self.ids[idx]}.npy')
        frame_num = self.id2frame_num[self.ids[idx]]
        if frame_num < 32:
            video_matrix[frame_num:] = 0.

        return {
            'title_seq': torch.tensor(title_seq, dtype=torch.long),
            'title_length': torch.tensor(title_length, dtype=torch.long),
            'video_matrix': torch.tensor(video_matrix, dtype=torch.float),
            'video_length': torch.tensor(frame_num, dtype=torch.long),
        }

########################################################################################################################
##### Train fn
def pointwise_train_fn(data_loader, model, optimizer, loss_fn, device, scheduler=None):
    model.train()
    losses = AverageMeter()

    tk0 = tqdm(data_loader, total=len(data_loader), desc="Train")
    for bi, d in enumerate(tk0):
        torch.cuda.empty_cache()
        title_seq = d["title_seq"].to(device, dtype=torch.long)
        title_length = d["title_length"].to(device, dtype=torch.long)
        video = d["video"].to(device, dtype=torch.float)
        video_length = d["video_length"].to(device, dtype=torch.long)
        labels = d["label"].to(device, dtype=torch.float)

        model.zero_grad()
        outputs, _ = model(title_seq=title_seq, title_length=title_length, video_matrix=video, video_length=video_length)
        # print('outputs: ', outputs.shape)
        # print('labels: ', outputs.shape)
        loss = loss_fn(outputs, labels) * 100
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()
        torch.cuda.empty_cache()

        losses.update(loss.item(), labels.size(0))
        tk0.set_postfix(loss=losses.avg)
        if DEBUG:
            if bi >= 10:
                break

def pointwise_infer_fn(data_loader, model, device):
    model.eval()

    pred_list, emb_pred_dict = [], {}
    # ema.apply_shadow()
    with torch.no_grad():
        tk0 = tqdm(data_loader, total=len(data_loader), desc="Pred")
        for bi, d in enumerate(tk0):
            torch.cuda.empty_cache()
            id_list = d['id']
            title_seq = d["title_seq"].to(device, dtype=torch.long)
            title_length = d["title_length"].to(device, dtype=torch.long)
            video = d["video"].to(device, dtype=torch.float)
            video_length = d["video_length"].to(device, dtype=torch.long)

            outputs, emb_outputs = model(title_seq=title_seq, title_length=title_length, video_matrix=video, video_length=video_length)
            # outputs = torch.sigmoid(outputs).cpu().detach().numpy().astype(float)
            outputs = outputs.cpu().detach().numpy().astype(float)
            emb_outputs = emb_outputs.cpu().detach().numpy().astype(float)
            pred_list.append(outputs)
            for _id, _emb in zip(id_list, emb_outputs):
                emb_pred_dict[_id] = _emb.tolist()
                # print('_id: ', _id)
                # print('_emb: ', _emb)

    return np.concatenate(pred_list).reshape(-1, len(dataset.tag2id)), emb_pred_dict

def pairwise_train_fn(data_loader, model, optimizer, loss_fn, device, scheduler=None, debug=False):
    model.train()
    losses = AverageMeter()

    tk0 = tqdm(data_loader, total=len(data_loader), desc="Train")
    for bi, d in enumerate(tk0):
        torch.cuda.empty_cache()
        title_seq1 = d["title_seq1"].to(device, dtype=torch.long)
        title_length1 = d["title_length1"].to(device, dtype=torch.long)
        video_matrix1 = d["video_matrix1"].to(device, dtype=torch.float)
        video_length1 = d["video_length1"].to(device, dtype=torch.long)
        title_seq2 = d["title_seq2"].to(device, dtype=torch.long)
        title_length2 = d["title_length2"].to(device, dtype=torch.long)
        video_matrix2 = d["video_matrix2"].to(device, dtype=torch.float)
        video_length2 = d["video_length2"].to(device, dtype=torch.long)
        labels = d["label"].to(device, dtype=torch.float)

        model.zero_grad()
        outputs = model(title_seq1, title_length1, video_matrix1, video_length1, title_seq2, title_length2, video_matrix2, video_length2)
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
            title_seq1 = d["title_seq1"].to(device, dtype=torch.long)
            title_length1 = d["title_length1"].to(device, dtype=torch.long)
            video_matrix1 = d["video_matrix1"].to(device, dtype=torch.float)
            video_length1 = d["video_length1"].to(device, dtype=torch.long)
            title_seq2 = d["title_seq2"].to(device, dtype=torch.long)
            title_length2 = d["title_length2"].to(device, dtype=torch.long)
            video_matrix2 = d["video_matrix2"].to(device, dtype=torch.float)
            video_length2 = d["video_length2"].to(device, dtype=torch.long)

            outputs = model(title_seq1, title_length1, video_matrix1, video_length1, title_seq2, title_length2, video_matrix2, video_length2)
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
            title_seq = d["title_seq"].to(device, dtype=torch.long)
            title_length = d["title_length"].to(device, dtype=torch.long)
            video_matrix = d["video_matrix"].to(device, dtype=torch.float)
            video_length = d["video_length"].to(device, dtype=torch.long)

            outputs = model.module.encoder(title_seq, title_length, video_matrix, video_length)
            # outputs = torch.sigmoid(outputs).cpu().detach().numpy().astype(float)
            outputs = outputs.cpu().detach().numpy().astype(float)
            pred_list.append(outputs)

    return np.concatenate(pred_list).reshape(-1, encode_dim)

def pointwise_loss_fn(output, label):
    # loss_fct = nn.MSELoss()
    # loss_fct = nn.BCEWithLogitsLoss()
    # loss_fct = nn.BCELoss()
    loss_fct = FocalBCELoss(gamma=2, alpha=0.99)
    return loss_fct(output, label)

def pairwise_loss_fn(output, label):
    loss_fct = nn.MSELoss()
    # loss_fct = nn.L1Loss()
    # loss_fct = nn.SmoothL1Loss()
    # loss_fct = nn.BCEWithLogitsLoss()
    # loss_fct = nn.BCELoss()
    # loss_fct = FocalBCELoss()
    # loss_fct = SpearmanLoss()
    return loss_fct(output, label)

########################################################################################################################
##### Model fn

def get_mask(sequences_batch, sequences_lengths):
    """
    Get the mask for a batch of padded variable length sequences.
    Args:
        sequences_batch: A batch of padded variable length sequences
            containing word indices. Must be a 2-dimensional tensor of size
            (batch, sequence).
        sequences_lengths: A tensor containing the lengths of the sequences in
            'sequences_batch'. Must be of size (batch).
    Returns:
        A mask of size (batch, max_sequence_length), where max_sequence_length
        is the length of the longest sequence in the batch.
    """
    batch_size = sequences_batch.size()[0]
    max_length = torch.max(sequences_lengths)
    mask = torch.ones(batch_size, max_length, dtype=torch.float)
    # mask[sequences_batch[:, :max_length] == 0] = 0.0
    return mask

# def get_mask(sequences_batch, sequences_lengths):
#     """
#     Get the mask for a batch of padded variable length sequences.
#     Args:
#         sequences_batch: A batch of padded variable length sequences
#             containing word indices. Must be a 2-dimensional tensor of size
#             (batch, sequence).
#         sequences_lengths: A tensor containing the lengths of the sequences in
#             'sequences_batch'. Must be of size (batch).
#     Returns:
#         A mask of size (batch, max_sequence_length), where max_sequence_length
#         is the length of the longest sequence in the batch.
#     """
#     max_length = torch.max(sequences_lengths)
#     mask = torch.ones(sequences_batch.size(), dtype=torch.float)
#     mask[sequences_batch[:, :max_length] == 0] = 0.0
#     return mask

def replace_masked(tensor, mask, value):
    """
    Replace the all the values of vectors in 'tensor' that are masked in
    'masked' by 'value'.
    Args:
        tensor: The tensor in which the masked vectors must have their values
            replaced.
        mask: A mask indicating the vectors which must have their values
            replaced.
        value: The value to place in the masked vectors of 'tensor'.
    Returns:
        A new tensor of the same size as 'tensor' where the values of the
        vectors masked in 'mask' were replaced by 'value'.
    """
    mask = mask.unsqueeze(1).transpose(2, 1)
    reverse_mask = 1.0 - mask
    values_to_add = value * reverse_mask
    return tensor * mask + values_to_add

def _init_esim_weights(module):
    """
    Initialise the weights of the ESIM model.
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight.data)
        nn.init.constant_(module.bias.data, 0.0)

    elif isinstance(module, nn.LSTM):
        nn.init.xavier_uniform_(module.weight_ih_l0.data)
        nn.init.orthogonal_(module.weight_hh_l0.data)
        nn.init.constant_(module.bias_ih_l0.data, 0.0)
        nn.init.constant_(module.bias_hh_l0.data, 0.0)
        hidden_size = module.bias_hh_l0.data.shape[0] // 4
        module.bias_hh_l0.data[hidden_size:(2*hidden_size)] = 1.0

        if (module.bidirectional):
            nn.init.xavier_uniform_(module.weight_ih_l0_reverse.data)
            nn.init.orthogonal_(module.weight_hh_l0_reverse.data)
            nn.init.constant_(module.bias_ih_l0_reverse.data, 0.0)
            nn.init.constant_(module.bias_hh_l0_reverse.data, 0.0)
            module.bias_hh_l0_reverse.data[hidden_size:(2*hidden_size)] = 1.0

class PointClfModel(nn.Module):
    def __init__(self, emb_dim, head_num, attention_layers):
        super(PointClfModel, self).__init__()
        self.WordEmb = nn.Embedding(len(word2id), W2V_DIM)
        self.WordEmb.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.WordEmb.weight.requires_grad = True
        self._rnn_dropout = RNNDropout(0.2)

        hidden_size = 1280
        self.lstm_encoder = Seq2SeqEncoder(nn.LSTM, W2V_DIM, hidden_size, bidirectional=True)
        self.video_linear = nn.Linear(dataset.video_shape[1], W2V_DIM)

        self._attention = SoftmaxAttention()

        self._projection = nn.Sequential(nn.Linear(4 * 2 * hidden_size, hidden_size),
                                         nn.ReLU())

        self.lstm_composition = Seq2SeqEncoder(nn.LSTM, hidden_size, hidden_size, bidirectional=True)

        self.encode_linear = nn.Linear(2 * 4 * hidden_size, emb_dim)
        self.clf_linear = nn.Linear(emb_dim, len(dataset.tag2id))

        self.dropout = nn.Dropout(0.3)

        # Initialize all weights and biases in the model.
        self.apply(_init_esim_weights)

    def encoder(self, title_seq, title_length, video_matrix, video_length):
        # self.lstm_encoder.flatten_parameters()
        # self.lstm_composition.flatten_parameters()

        title_mask = get_mask(title_seq, title_length).to(device)
        video_mask = get_mask(video_matrix, video_length).to(device)

        title_emb = self.WordEmb(title_seq)
        title_emb = self._rnn_dropout(title_emb)
        title_lstm = self.lstm_encoder(title_emb, title_length)

        video_emb = self.video_linear(video_matrix)
        video_emb = self._rnn_dropout(video_emb)
        video_lstm = self.lstm_encoder(video_emb, video_length)

        title_soft_att, video_soft_att = self._attention(title_lstm, title_mask, video_lstm, video_mask)

        title_enhanced = torch.cat([title_lstm, title_soft_att, title_lstm - title_soft_att, title_lstm * title_soft_att], dim=-1)
        video_enhanced = torch.cat([video_lstm, video_soft_att, video_lstm - video_soft_att, video_lstm * video_soft_att], dim=-1)

        title_projected = self._projection(title_enhanced)
        video_projected = self._projection(video_enhanced)

        title_projected = self._rnn_dropout(title_projected)
        video_projected = self._rnn_dropout(video_projected)

        v_ai = self.lstm_composition(title_projected, title_length)
        v_bj = self.lstm_composition(video_projected, video_length)

        v_a_avg = torch.sum(v_ai * title_mask.unsqueeze(1)
                            .transpose(2, 1), dim=1) \
                  / torch.sum(title_mask, dim=1, keepdim=True)
        v_b_avg = torch.sum(v_bj * video_mask.unsqueeze(1)
                            .transpose(2, 1), dim=1) \
                  / torch.sum(video_mask, dim=1, keepdim=True)

        v_a_max, _ = replace_masked(v_ai, title_mask, -1e7).max(dim=1)
        v_b_max, _ = replace_masked(v_bj, video_mask, -1e7).max(dim=1)

        concat = torch.cat([v_a_avg, v_a_max, v_b_avg, v_b_max], dim=1)

        encoded_output = self.encode_linear(concat)
        return encoded_output

    def forward(self, title_seq, title_length, video_matrix, video_length):
        encoded_emb = self.encoder(title_seq, title_length, video_matrix, video_length)
        normed_embedding = torch.nn.functional.normalize(encoded_emb, p=2, dim=1)
        logits = self.clf_linear(encoded_emb)
        logits = torch.sigmoid(logits)

        return logits, normed_embedding

class PairSiameseModel(nn.Module):
    def __init__(self, point_encode_model, emb_dim):
        super(PairSiameseModel, self).__init__()
        self.pointwise_module = point_encode_model

        self.clf_linear = nn.Linear(emb_dim, len(dataset.tag2id))

        self.dropout = nn.Dropout(0.3)

    def encoder(self, title_seq, title_length, video_matrix, video_length):
        # encoded_output = self.dropout(feat)
        # encoded_output = torch.relu(encoded_output)
        encoded_output = self.pointwise_module.module.encoder(title_seq, title_length, video_matrix, video_length)
        encoded_output = torch.nn.functional.normalize(encoded_output, p=2, dim=1)
        return encoded_output

    def forward(self, title_seq1, title_length1, video_matrix1, video_length1, title_seq2, title_length2, video_matrix2, video_length2):
        encoded_emb1 = self.encoder(title_seq1, title_length1, video_matrix1, video_length1)
        encoded_emb2 = self.encoder(title_seq2, title_length2, video_matrix2, video_length2)
        return torch.cosine_similarity(encoded_emb1, encoded_emb2)

########################################################################################################################
##### Run
seed_everything(seed=42)
dataset = LoadDataset(data_path=DATA_PATH, debug=DEBUG)

word2tag_feat = load_pickle('../data/new_data/word_feat/word2tag_feat.pkl')

# dataset.df['title_seg'] = dataset.df['title'].apply(lambda x: jieba.lcut(x))
dataset.df['title_seg'] = parallelize_df_func(dataset.df['title'], jieba_seg)
word2emb = load_pickle(f'{W2V_PATH}/w2v_emb_new.pkl')

word_list = list(word2emb.keys())
print('\nword_list: ', len(word_list))

word2id = {}
embedding_matrix = np.zeros((len(word_list)+1, W2V_DIM))
word_tag_embedding_matrix = np.zeros((len(word_list)+1, 10000))
for idx, word in enumerate(word_list):
    word2id[word] = idx+1
    embedding_matrix[idx+1] = word2emb[word]
    if word in word2tag_feat:
        word_tag_embedding_matrix[idx+1] = word2tag_feat[word]
id2seq = {}
for _id, _title_seg in dataset.df[['id', 'title_seg']].values:
    seq = []
    for word in _title_seg:
        if word in word2id:
            seq.append(word2id[word])
    id2seq[str(_id)] = seq
save_pickle(word2id, f'{SAVE_PATH}/word2id.pkl')

result_dict = {}
##### Pointwise train
if POINTWISE_TRAIN:
    print('\nStart Pointwise Train:')
    GPU_NUM = 1
    BATCH_SIZE = 1024 * GPU_NUM
    MAX_TITLE_SEQ = 32
    EPOCHS = 100
    LR = 1e-3
    WARMUP = 0.05
    WEIGHT_DECAY = 0.
    EARLYSTOP_NUM = 5
    SCHEDULE_DECAY = 0

    if DEBUG:
        EPOCHS = 1

    train_set = PointDataset(id_list=dataset.pointwise_ids+dataset.pairwise_ids, dataset=dataset, id2seq=id2seq, max_len=MAX_TITLE_SEQ)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    valid_set = PointDataset(id_list=dataset.pairwise_ids, dataset=dataset, id2seq=id2seq, max_len=MAX_TITLE_SEQ)
    valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_set = PointDataset(id_list=dataset.test_ids, dataset=dataset, id2seq=id2seq, max_len=MAX_TITLE_SEQ, is_test=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    torch.cuda.empty_cache()
    device = torch.device("cuda:0")
    model = PointClfModel(ENCODE_DIM, head_num=HEAD_NUM, attention_layers=LAYER_NUM)
    model.to(device)
    model = nn.DataParallel(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=SCHEDULE_DECAY, verbose=True)
    best_val_score = 0.
    es = EarlyStopping(patience=EARLYSTOP_NUM, mode="max")
    MODEL_WEIGHT = f"{SAVE_PATH}/pointwise_model.bin"
    for epoch in range(EPOCHS):
        pointwise_train_fn(train_loader, model, optimizer, pointwise_loss_fn, device, scheduler=None)
        valid_pred, valid_emb_pred = pointwise_infer_fn(valid_loader, model, device)
        valid_spearmanr = caculate_spearmanr_score(pair_df=dataset.label_df, emb_dict=valid_emb_pred)
        print(f'Epoch {epoch + 1}/{EPOCHS} valid_spearmanr={valid_spearmanr:.4f}')
        if valid_spearmanr >= best_val_score:
            best_val_score = valid_spearmanr
        scheduler.step(valid_spearmanr)
        es(valid_spearmanr, model, model_path=MODEL_WEIGHT)
        if es.early_stop:
            print("Early stopping")
            break

    result_dict['pointwise'] = best_val_score
    pd.DataFrame([result_dict]).to_csv(f'{SAVE_PATH}/result_log.csv', index=False)
    # if not DEBUG:
    if True:
        model.load_state_dict(torch.load(MODEL_WEIGHT))
        _, valid_emb_pred = pointwise_infer_fn(valid_loader, model, device)
        _, test_emb_pred = pointwise_infer_fn(test_loader, model, device)
        save_json(f'{SAVE_PATH}/pointwise_valid_pred.json', valid_emb_pred)
        save_json(f'{SAVE_PATH}/pointwise_test_emb_pred.json', test_emb_pred)

##### Pairwise train
if PAIRWISE_TRAIN:
    print('\nStart Pairwise Train:')
    GPU_NUM = 1
    NFOLDS = 5
    BATCH_SIZE = 512 * GPU_NUM
    MAX_TITLE_SEQ = 32
    EPOCHS = 100
    LR = 1e-3
    WARMUP = 0.05
    WEIGHT_DECAY = 0.
    EARLYSTOP_NUM = 5
    SCHEDULE_DECAY = 1

    if DEBUG:
        EPOCHS = 1
        NFOLDS = 2

    oof_pred = np.zeros(len(dataset.label_df))
    pairwise_pred = np.zeros((len(dataset.pairwise_ids), 256))
    test_pred = np.zeros((len(dataset.test_ids), ENCODE_DIM))
    cv_score = []
    pairwise_set = EmbDataset(id_list=dataset.pairwise_ids, dataset=dataset, id2seq=id2seq, max_len=MAX_TITLE_SEQ)
    pairwise_loader = DataLoader(pairwise_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_set = EmbDataset(id_list=dataset.test_ids, dataset=dataset, id2seq=id2seq, max_len=MAX_TITLE_SEQ)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    start_time = time.time()
    kf = KFold(n_splits=NFOLDS, shuffle=True, random_state=42)
    for _fold, (train_idx, valid_idx) in enumerate(kf.split(dataset.label_df, dataset.label_df)):
        seed_everything(seed=42+_fold)
        print(f'\nFold{_fold}: {len(train_idx)}|{len(valid_idx)}')

        train_set = PairDataset(df=dataset.label_df.iloc[train_idx], dataset=dataset, id2seq=id2seq, max_len=MAX_TITLE_SEQ)
        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
        valid_set = PairDataset(df=dataset.label_df.iloc[valid_idx], dataset=dataset, id2seq=id2seq, max_len=MAX_TITLE_SEQ)
        valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
        # for batch in train_loader:
        #     print(batch)
        #     break

        torch.cuda.empty_cache()
        device = torch.device("cuda:0")
        point_model = PointClfModel(emb_dim=ENCODE_DIM, head_num=HEAD_NUM, attention_layers=LAYER_NUM)
        point_model = nn.DataParallel(point_model)
        point_model.load_state_dict(torch.load(f'{SAVE_PATH}/pointwise_model.bin'))

        model = PairSiameseModel(point_model, emb_dim=ENCODE_DIM)
        model.to(device)
        model = nn.DataParallel(model)

        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        scheduler = None

        es = EarlyStopping(patience=EARLYSTOP_NUM, mode="max")
        MODEL_WEIGHT = f"{SAVE_PATH}/pairwise_model{_fold}.bin"
        for epoch in range(EPOCHS):
            pairwise_train_fn(train_loader, model, optimizer, pairwise_loss_fn, device, scheduler=scheduler, debug=DEBUG)
            valid_pred = pairwise_infer_fn(valid_loader, model, device)
            valid_spearmanr = scipy.stats.spearmanr(valid_pred, dataset.label_df.loc[valid_idx].label.values).correlation

            print(f'Epoch {epoch + 1}/{EPOCHS} valid_spearmanr={valid_spearmanr:.4f}')

            es(valid_spearmanr, model, model_path=MODEL_WEIGHT)
            if es.early_stop:
                print("Early stopping")
                break
        model.load_state_dict(torch.load(MODEL_WEIGHT))
        oof_pred[valid_idx] = pairwise_infer_fn(valid_loader, model, device)
        valid_spearmanr = scipy.stats.spearmanr(oof_pred[valid_idx], dataset.label_df.loc[valid_idx].label.values).correlation
        cv_score.append(valid_spearmanr)

        pairwise_pred += pairwise_emb_infer_fn(pairwise_loader, model, ENCODE_DIM, device) / NFOLDS
        test_pred += pairwise_emb_infer_fn(test_loader, model, ENCODE_DIM, device) / NFOLDS

    oof_spearmanr = scipy.stats.spearmanr(oof_pred, dataset.label_df.label.values).correlation
    cv_spearmanr = np.mean(cv_score)
    print(f'oof_spearmanr={oof_spearmanr:.4f} cv_spearmanr={cv_spearmanr:.4f}')

    result_dict['pairwise'] = oof_spearmanr
    pd.DataFrame([result_dict]).to_csv(f'{SAVE_PATH}/result_log.csv', index=False)
    np.save(f'{SAVE_PATH}/pairwise_pairwise_pred.npy', pairwise_pred)
    np.save(f'{SAVE_PATH}/pairwise_test_pred.npy', test_pred)
    # if not DEBUG:
    if True:
        emb_pred = {}
        for vid, emb in zip(dataset.test_ids, test_pred):
            emb_pred[str(vid)] = emb.tolist()
        with open('result.json', 'w') as f:
            json.dump(emb_pred, f)
        with ZipFile(f'{SAVE_PATH}/paiwise_sub.zip', 'w', compression=ZIP_DEFLATED) as zip_file:
            zip_file.write('result.json')

    # if not DEBUG:
    if True:
        print('\tPointwise emb infering...')
        pointwise_pred = np.zeros((len(dataset.pointwise_ids), 256))
        pointwise_set = EmbDataset(id_list=dataset.pointwise_ids, dataset=dataset, id2seq=id2seq, max_len=MAX_TITLE_SEQ)
        pointwise_loader = DataLoader(pointwise_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

        for _fold in range(NFOLDS):
            torch.cuda.empty_cache()
            device = torch.device("cuda:0")
            point_model = PointClfModel(emb_dim=ENCODE_DIM, head_num=HEAD_NUM, attention_layers=LAYER_NUM)
            point_model = nn.DataParallel(point_model)
            point_model.load_state_dict(torch.load(f'{SAVE_PATH}/pointwise_model.bin'))

            model = PairSiameseModel(point_model, emb_dim=ENCODE_DIM)
            model.to(device)
            model = nn.DataParallel(model)
            MODEL_WEIGHT = f"{SAVE_PATH}/pairwise_model{_fold}.bin"
            model.load_state_dict(torch.load(MODEL_WEIGHT))

            pointwise_pred += pairwise_emb_infer_fn(pointwise_loader, model, ENCODE_DIM, device) / NFOLDS
        np.save(f'{SAVE_PATH}/pairwise_pointwise_pred.npy', pointwise_pred)

##### Distill train
if DISTILL_TRAIN:
    print('\nStart Distill Train:')
    GPU_NUM = 1
    NFOLDS = 5
    BATCH_SIZE = 512 * GPU_NUM
    MAX_TITLE_SEQ = 32
    EPOCHS = 100
    LR = 1e-3
    WARMUP = 0.05
    WEIGHT_DECAY = 0.
    EARLYSTOP_NUM = 5
    SCHEDULE_DECAY = 1

    if DEBUG:
        EPOCHS = 1
        NFOLDS = 2

    print('\tGet softlabel')
    pairwise_emb = np.load(f'{SAVE_PATH}/pairwise_pairwise_pred.npy')
    id2emb = {}
    for id, emb in zip(dataset.pairwise_ids, pairwise_emb):
        id2emb[id] = emb
    softlabel_list = []
    for id1, id2 in dataset.label_df[['id1', 'id2']].values:
        _pred = cosine_similarity([id2emb[str(id1)]], [id2emb[str(id2)]])[0][0]
        softlabel_list.append(_pred)
    label_df = dataset.label_df.copy(deep=True)
    label_df['softlabel'] = softlabel_list
    label_df['gt_label'] = label_df['label'].values
    label_df['label'] = label_df['label'] * 0.7 + label_df['softlabel'] * 0.3

    oof_pred = np.zeros(len(label_df))
    pairwise_pred = np.zeros((len(dataset.pairwise_ids), 256))
    test_pred = np.zeros((len(dataset.test_ids), ENCODE_DIM))
    cv_score = []
    pairwise_set = EmbDataset(id_list=dataset.pairwise_ids, dataset=dataset, id2seq=id2seq, max_len=MAX_TITLE_SEQ)
    pairwise_loader = DataLoader(pairwise_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_set = EmbDataset(id_list=dataset.test_ids, dataset=dataset, id2seq=id2seq, max_len=MAX_TITLE_SEQ)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    start_time = time.time()
    kf = KFold(n_splits=NFOLDS, shuffle=True, random_state=42)
    for _fold, (train_idx, valid_idx) in enumerate(kf.split(label_df, label_df)):
        seed_everything(seed=42 + _fold)
        print(f'\nFold{_fold}: {len(train_idx)}|{len(valid_idx)}')

        train_set = PairDataset(df=label_df.iloc[train_idx], dataset=dataset, id2seq=id2seq,
                                max_len=MAX_TITLE_SEQ)
        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
        valid_set = PairDataset(df=label_df.iloc[valid_idx], dataset=dataset, id2seq=id2seq,
                                max_len=MAX_TITLE_SEQ)
        valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
        # for batch in train_loader:
        #     print(batch)
        #     break

        torch.cuda.empty_cache()
        device = torch.device("cuda:0")
        point_model = PointClfModel(emb_dim=ENCODE_DIM, head_num=HEAD_NUM, attention_layers=LAYER_NUM)
        point_model = nn.DataParallel(point_model)
        point_model.load_state_dict(torch.load(f'{SAVE_PATH}/pointwise_model.bin'))

        model = PairSiameseModel(point_model, emb_dim=ENCODE_DIM)
        model.to(device)
        model = nn.DataParallel(model)

        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        scheduler = None

        es = EarlyStopping(patience=EARLYSTOP_NUM, mode="max")
        MODEL_WEIGHT = f"{SAVE_PATH}/distill_model{_fold}.bin"
        for epoch in range(EPOCHS):
            pairwise_train_fn(train_loader, model, optimizer, pairwise_loss_fn, device, scheduler=scheduler,
                              debug=DEBUG)
            valid_pred = pairwise_infer_fn(valid_loader, model, device)
            valid_spearmanr = scipy.stats.spearmanr(valid_pred, label_df.loc[valid_idx].label.values).correlation

            print(f'Epoch {epoch + 1}/{EPOCHS} valid_spearmanr={valid_spearmanr:.4f}')

            es(valid_spearmanr, model, model_path=MODEL_WEIGHT)
            if es.early_stop:
                print("Early stopping")
                break
        model.load_state_dict(torch.load(MODEL_WEIGHT))
        oof_pred[valid_idx] = pairwise_infer_fn(valid_loader, model, device)
        valid_spearmanr = scipy.stats.spearmanr(oof_pred[valid_idx], label_df.loc[valid_idx].label.values).correlation
        cv_score.append(valid_spearmanr)

        pairwise_pred += pairwise_emb_infer_fn(pairwise_loader, model, ENCODE_DIM, device) / NFOLDS
        test_pred += pairwise_emb_infer_fn(test_loader, model, ENCODE_DIM, device) / NFOLDS

    oof_spearmanr = scipy.stats.spearmanr(oof_pred, label_df.gt_label.values).correlation
    cv_spearmanr = np.mean(cv_score)
    print(f'oof_spearmanr={oof_spearmanr:.4f} cv_spearmanr={cv_spearmanr:.4f}')

    result_dict['distill'] = oof_spearmanr
    pd.DataFrame([result_dict]).to_csv(f'{SAVE_PATH}/result_log.csv', index=False)
    np.save(f'{SAVE_PATH}/distill_pairwise_pred.npy', pairwise_pred)
    np.save(f'{SAVE_PATH}/distill_test_pred.npy', test_pred)
    # if not DEBUG:
    if True:
        emb_pred = {}
        for vid, emb in zip(dataset.test_ids, test_pred):
            emb_pred[str(vid)] = emb.tolist()
        with open('result.json', 'w') as f:
            json.dump(emb_pred, f)
        with ZipFile(f'{SAVE_PATH}/distill_sub.zip', 'w', compression=ZIP_DEFLATED) as zip_file:
            zip_file.write('result.json')

    # if not DEBUG:
    if True:
        print('\tPointwise emb infering...')
        pointwise_pred = np.zeros((len(dataset.pointwise_ids), 256))
        pointwise_set = EmbDataset(id_list=dataset.pointwise_ids, dataset=dataset, id2seq=id2seq, max_len=MAX_TITLE_SEQ)
        pointwise_loader = DataLoader(pointwise_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

        for _fold in range(NFOLDS):
            torch.cuda.empty_cache()
            device = torch.device("cuda:0")
            point_model = PointClfModel(emb_dim=ENCODE_DIM, head_num=HEAD_NUM, attention_layers=LAYER_NUM)
            point_model = nn.DataParallel(point_model)
            point_model.load_state_dict(torch.load(f'{SAVE_PATH}/pointwise_model.bin'))

            model = PairSiameseModel(point_model, emb_dim=ENCODE_DIM)
            model.to(device)
            model = nn.DataParallel(model)
            MODEL_WEIGHT = f"{SAVE_PATH}/distill_model{_fold}.bin"
            model.load_state_dict(torch.load(MODEL_WEIGHT))

            pointwise_pred += pairwise_emb_infer_fn(pointwise_loader, model, ENCODE_DIM, device) / NFOLDS
        np.save(f'{SAVE_PATH}/distill_pointwise_pred.npy', pointwise_pred)

##### Distill2 train
if DISTILL_TRAIN2:
    print('\nStart Distill2 Train:')
    GPU_NUM = 1
    NFOLDS = 5
    BATCH_SIZE = 512 * GPU_NUM
    MAX_TITLE_SEQ = 32
    EPOCHS = 100
    LR = 1e-3
    WARMUP = 0.05
    WEIGHT_DECAY = 0.
    EARLYSTOP_NUM = 5
    SCHEDULE_DECAY = 1

    if DEBUG:
        EPOCHS = 1
        NFOLDS = 2

    print('\tGet softlabel')
    pairwise_emb = np.load(f'{SAVE_PATH}/distill_pairwise_pred.npy')
    id2emb = {}
    for id, emb in zip(dataset.pairwise_ids, pairwise_emb):
        id2emb[id] = emb
    softlabel_list = []
    for id1, id2 in dataset.label_df[['id1', 'id2']].values:
        _pred = cosine_similarity([id2emb[str(id1)]], [id2emb[str(id2)]])[0][0]
        softlabel_list.append(_pred)
    label_df = dataset.label_df.copy(deep=True)
    label_df['softlabel'] = softlabel_list
    label_df['gt_label'] = label_df['label'].values
    label_df['label'] = label_df['label'] * 0.7 + label_df['softlabel'] * 0.3

    oof_pred = np.zeros(len(label_df))
    pairwise_pred = np.zeros((len(dataset.pairwise_ids), 256))
    test_pred = np.zeros((len(dataset.test_ids), ENCODE_DIM))
    cv_score = []
    pairwise_set = EmbDataset(id_list=dataset.pairwise_ids, dataset=dataset, id2seq=id2seq, max_len=MAX_TITLE_SEQ)
    pairwise_loader = DataLoader(pairwise_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_set = EmbDataset(id_list=dataset.test_ids, dataset=dataset, id2seq=id2seq, max_len=MAX_TITLE_SEQ)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    start_time = time.time()
    kf = KFold(n_splits=NFOLDS, shuffle=True, random_state=42)
    for _fold, (train_idx, valid_idx) in enumerate(kf.split(label_df, label_df)):
        seed_everything(seed=42 + _fold)
        print(f'\nFold{_fold}: {len(train_idx)}|{len(valid_idx)}')

        train_set = PairDataset(df=label_df.iloc[train_idx], dataset=dataset, id2seq=id2seq,
                                max_len=MAX_TITLE_SEQ)
        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
        valid_set = PairDataset(df=label_df.iloc[valid_idx], dataset=dataset, id2seq=id2seq,
                                max_len=MAX_TITLE_SEQ)
        valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
        # for batch in train_loader:
        #     print(batch)
        #     break

        torch.cuda.empty_cache()
        device = torch.device("cuda:0")
        point_model = PointClfModel(emb_dim=ENCODE_DIM, head_num=HEAD_NUM, attention_layers=LAYER_NUM)
        point_model = nn.DataParallel(point_model)
        point_model.load_state_dict(torch.load(f'{SAVE_PATH}/pointwise_model.bin'))

        model = PairSiameseModel(point_model, emb_dim=ENCODE_DIM)
        model.to(device)
        model = nn.DataParallel(model)

        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        scheduler = None

        es = EarlyStopping(patience=EARLYSTOP_NUM, mode="max")
        MODEL_WEIGHT = f"{SAVE_PATH}/distill2_model{_fold}.bin"
        for epoch in range(EPOCHS):
            pairwise_train_fn(train_loader, model, optimizer, pairwise_loss_fn, device, scheduler=scheduler,
                              debug=DEBUG)
            valid_pred = pairwise_infer_fn(valid_loader, model, device)
            valid_spearmanr = scipy.stats.spearmanr(valid_pred, label_df.loc[valid_idx].label.values).correlation

            print(f'Epoch {epoch + 1}/{EPOCHS} valid_spearmanr={valid_spearmanr:.4f}')

            es(valid_spearmanr, model, model_path=MODEL_WEIGHT)
            if es.early_stop:
                print("Early stopping")
                break
        model.load_state_dict(torch.load(MODEL_WEIGHT))
        oof_pred[valid_idx] = pairwise_infer_fn(valid_loader, model, device)
        valid_spearmanr = scipy.stats.spearmanr(oof_pred[valid_idx], label_df.loc[valid_idx].label.values).correlation
        cv_score.append(valid_spearmanr)

        pairwise_pred += pairwise_emb_infer_fn(pairwise_loader, model, ENCODE_DIM, device) / NFOLDS
        test_pred += pairwise_emb_infer_fn(test_loader, model, ENCODE_DIM, device) / NFOLDS

    oof_spearmanr = scipy.stats.spearmanr(oof_pred, label_df.gt_label.values).correlation
    cv_spearmanr = np.mean(cv_score)
    print(f'oof_spearmanr={oof_spearmanr:.4f} cv_spearmanr={cv_spearmanr:.4f}')

    result_dict['distill2'] = oof_spearmanr
    pd.DataFrame([result_dict]).to_csv(f'{SAVE_PATH}/result_log.csv', index=False)
    np.save(f'{SAVE_PATH}/distill2_pairwise_pred.npy', pairwise_pred)
    np.save(f'{SAVE_PATH}/distill2_test_pred.npy', test_pred)
    # if not DEBUG:
    if True:
        emb_pred = {}
        for vid, emb in zip(dataset.test_ids, test_pred):
            emb_pred[str(vid)] = emb.tolist()
        with open('result.json', 'w') as f:
            json.dump(emb_pred, f)
        with ZipFile(f'{SAVE_PATH}/distill2_sub.zip', 'w', compression=ZIP_DEFLATED) as zip_file:
            zip_file.write('result.json')

    # if not DEBUG:
    if True:
        print('\tPointwise emb infering...')
        pointwise_pred = np.zeros((len(dataset.pointwise_ids), 256))
        pointwise_set = EmbDataset(id_list=dataset.pointwise_ids, dataset=dataset, id2seq=id2seq, max_len=MAX_TITLE_SEQ)
        pointwise_loader = DataLoader(pointwise_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

        for _fold in range(NFOLDS):
            torch.cuda.empty_cache()
            device = torch.device("cuda:0")
            point_model = PointClfModel(emb_dim=ENCODE_DIM, head_num=HEAD_NUM, attention_layers=LAYER_NUM)
            point_model = nn.DataParallel(point_model)
            point_model.load_state_dict(torch.load(f'{SAVE_PATH}/pointwise_model.bin'))

            model = PairSiameseModel(point_model, emb_dim=ENCODE_DIM)
            model.to(device)
            model = nn.DataParallel(model)
            MODEL_WEIGHT = f"{SAVE_PATH}/distill2_model{_fold}.bin"
            model.load_state_dict(torch.load(MODEL_WEIGHT))

            pointwise_pred += pairwise_emb_infer_fn(pointwise_loader, model, ENCODE_DIM, device) / NFOLDS
        np.save(f'{SAVE_PATH}/distill2_pointwise_pred.npy', pointwise_pred)
