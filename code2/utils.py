import os
import json
import jieba
import scipy
import random
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
from sklearn.metrics.pairwise import cosine_similarity

import torch
import torch.nn as nn

TEST_FLAG = 'test_b'

class LoadDataset:
    def __init__(self, data_path, debug=False):
        self.data_path = data_path

        if debug:
            self.df = pd.read_csv(f'{data_path}/whole_df_new_debug.csv', sep='\t', encoding='gb18030')
        else:
            self.df = pd.read_csv(f'{data_path}/whole_df_new.csv', sep='\t', encoding='gb18030')
        print('*' * 20)
        print(self.df.flag.unique())
        print('*' * 20)
        # if debug:
        #     self.df = pd.concat([
        #         self.df.loc[self.df.flag == 'pointwise'].sample(n=10000),
        #         self.df.loc[self.df.flag == 'test_b'],
        #         self.df.loc[self.df.flag == 'pairwise'],
        #     ]).reset_index(drop=True)
        self.df['id'] = self.df['id'].astype('str')
        print(f'whole_df: {len(self.df)}')
        if debug:
            self.label_df = pd.read_csv(f'{data_path}/label_df_debug.csv', sep='\t')
        else:
            self.label_df = pd.read_csv(f'{data_path}/label.tsv', sep='\t', header=None)
        self.label_df.columns = ['id1', 'id2', 'label']
        self.label_df['id1'] = self.label_df['id1'].astype(str)
        self.label_df['id2'] = self.label_df['id2'].astype(str)
        print(f'label_df: {len(self.label_df)}')
        self.tag2id = self.get_tag2id()

        self.id2title, self.id2asr, self.id2frame_num, self.id2tag, self.id2cate = self.get_dict_info()
        print(f'id2title:{len(self.id2title)}')
        print(f'id2asr:{len(self.id2asr)}')
        print(f'id2frame_num:{len(self.id2frame_num)}')
        print(f'id2tag:{len(self.id2tag)}')
        print(f'id2cate:{len(self.id2cate)}')

        self.pointwise_ids = self.df.loc[self.df.flag == 'pointwise'].id.values.tolist()
        self.pairwise_ids = self.df.loc[self.df.flag == 'pairwise'].id.values.tolist()
        self.test_ids = self.df.loc[self.df.flag == TEST_FLAG].id.values.tolist()
        print(f'Pointwise|Pairwise|Test id nums={len(self.pointwise_ids)}|{len(self.pairwise_ids)}|{len(self.test_ids)}')

        self.video_shape = [32, 1536]

    def get_tag2id(self):
        with open(f'{self.data_path}/tag_list.txt', 'r') as f:
            tag_list = f.read().strip().split('\n')
        print('tag_num: ', len(tag_list))
        tag2id = {}
        for idx, tag in enumerate(tag_list):
            tag2id[tag] = idx
        return tag2id

    def get_dict_info(self):
        id2title = {}
        id2asr = {}
        id2frame_num = {}
        for _id, _title, _asr_text, _num_frames in tqdm(self.df[['id', 'title', 'asr_text', 'num_frames']].values, desc='Making Whole Id Dicts'):
            id2title[_id] = _title
            id2asr[_id] = _asr_text
            id2frame_num[_id] = _num_frames

        id2tag = {}
        id2cate = {}
        for _id, _tag, _category in tqdm(self.df.loc[~self.df.flag.isin(['test_a', 'test_b'])][['id', 'tag_id', 'category_id']].values, desc='Making Train Id Dicts'):
            id2tag[_id] = _tag
            id2cate[_id] = _category

        return id2title, id2asr, id2frame_num, id2tag, id2cate

def jieba_seg(df):
    return df.apply(lambda x: jieba.lcut(x))

def parallelize_df_func(df, func, num_partitions=16, n_jobs=8):
    df_split = np.array_split(df, num_partitions)
    pool = Pool(n_jobs)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def save_pickle(dic, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump(dic, f)

def load_pickle(load_path):
    with open(load_path, 'rb') as f:
        message_dict = pickle.load(f)
    return message_dict

def save_json(save_path, dic):
    with open(save_path, 'w') as f:
        json.dump(dic, f)

def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

class AverageMeter:
    """
    Computes and stores the average and current value
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class EarlyStopping:
    def __init__(self, patience=7, mode="max", delta=0.):
        self.patience = patience
        self.counter = 0
        self.mode = mode
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        if self.mode == "min":
            self.val_score = np.Inf
        else:
            self.val_score = -np.Inf

    def __call__(self, epoch_score, model, model_path):

        if self.mode == "min":
            score = -1.0 * epoch_score
        else:
            score = np.copy(epoch_score)

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
        elif score < self.best_score: #  + self.delta
            self.counter += 1
            print('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            # ema.apply_shadow()
            self.save_checkpoint(epoch_score, model, model_path)
            # ema.restore()
            self.counter = 0

    def save_checkpoint(self, epoch_score, model, model_path):
        if epoch_score not in [-np.inf, np.inf, -np.nan, np.nan]:
            print('Validation score improved ({} --> {}). Saving model!'.format(self.val_score, epoch_score))
            torch.save(model.state_dict(), model_path)
        self.val_score = epoch_score

def caculate_spearmanr_score(pair_df, emb_dict):
    y_true = pair_df.label.values
    y_pred = []
    for _id1, _id2 in pair_df[['id1', 'id2']].values:
        _pred = cosine_similarity([emb_dict[str(_id1)]], [emb_dict[str(_id2)]])[0][0]
        y_pred.append(_pred)
    return scipy.stats.spearmanr(y_pred, y_true).correlation

class FocalBCELoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.65, size_average=True):
        super(FocalBCELoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average

    def forward(self, logits, targets):
        l = logits.reshape(-1)
        t = targets.reshape(-1)
        p = torch.where(t >= 0.5, l, 1 - l)
        a = torch.where(t >= 0.5, self.alpha, 1 - self.alpha)
        logp = torch.log(torch.clamp(p, 1e-4, 1-1e-4))
        loss = - a * logp * ((1 - p)**self.gamma)
        if self.size_average: return loss.mean()
        else: return loss.sum()

class SpearmanLoss(nn.Module):
    def __init__(self, size_average=True):
        super(SpearmanLoss, self).__init__()
        self.size_average = size_average

    def forward(self, logits, targets):
        loss = torch.var(logits - targets, dim=-1) / torch.std(logits, dim=-1) * torch.std(targets, dim=-1)
        if self.size_average: return loss.mean()
        else: return loss.sum()