import os
import jieba
import gensim
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter
from multiprocessing import Pool

from utils import save_pickle, LoadDataset, parallelize_df_func

DEBUG = False

DATA_PATH = '../data/new_data/'
SAVE_PATH = f'{DATA_PATH}/word_feat'
if not os.path.exists(SAVE_PATH):
    os.mkdir(SAVE_PATH)

MIN_CNT = 3

if DEBUG:
    SAVE_PATH = f'{SAVE_PATH}/debug'
    if not os.path.exists(SAVE_PATH):
        os.mkdir(SAVE_PATH)

def jieba_seg(df):
    return df.apply(lambda x: jieba.lcut(x))

def chose_seg(df):
    return df.apply(lambda x: [word for word in x if word in chosen_word])

dataset = LoadDataset(data_path=DATA_PATH)

whole_df = dataset.df.loc[dataset.df.flag == 'pointwise'].reset_index(drop=True)
whole_df['title_seg'] = parallelize_df_func(whole_df['title'], jieba_seg)

word_cnt = Counter(list(itertools.chain(*whole_df.title_seg.values.tolist())))
word_cnt_df = pd.DataFrame()
word_cnt_df['word'] = word_cnt.keys()
word_cnt_df['cnt'] = word_cnt.values()
word_cnt_df = word_cnt_df.sort_values('cnt', ascending=False).reset_index(drop=True)

chosen_word_list = word_cnt_df.loc[word_cnt_df.cnt >= MIN_CNT].word.values.tolist()
print('\nchosen_word_list: ', len(chosen_word_list))
word2tag_feat = {}
for _idx, _word in enumerate(chosen_word_list):
    word2tag_feat[_word] = np.zeros(10000)

for _word_list, _tag_list in tqdm(whole_df.loc[whole_df.flag == 'pointwise'][['title_seg', 'tag_id']].values):
    _tag_list = str(_tag_list).split(' ')
    for word in _word_list:
        if word in word2tag_feat:
            for tag in _tag_list:
                if tag in dataset.tag2id:
                    word2tag_feat[word][dataset.tag2id[tag]] += 1
                    # print('**')
for word in word2tag_feat:
    word2tag_feat[word] /= word_cnt[word]

save_pickle(word2tag_feat, f'{SAVE_PATH}/word2tag_feat.pkl')

"""
whole_df['category_id'] = whole_df['category_id'].fillna(-100).astype(int)
whole_df['category_id2'] = whole_df['category_id'] // 100

cate2id1, cate2id2 = {}, {}
for idx, cate in enumerate(whole_df.category_id.unique()):
    cate2id1[cate] = idx
for idx, cate in enumerate(whole_df.category_id2.unique()):
    cate2id2[cate] = len(cate2id1) + idx
print('cate2id1: ', len(cate2id1))
print('cate2id2: ', len(cate2id2))

word2cate_feat = {}
for _idx, _word in enumerate(chosen_word_list):
    word2cate_feat[_word] = np.zeros(len(cate2id1)+len(cate2id2))

for _word_list, _cate1, _cate2 in tqdm(whole_df.loc[whole_df.flag == 'pointwise'][['title_seg', 'category_id', 'category_id2']].values):
    for word in _word_list:
        if word in word2cate_feat:
            word2cate_feat[word][cate2id1[_cate1]] += 1
            word2cate_feat[word][cate2id2[_cate2]] += 1
                    # print('**')
for word in word2cate_feat:
    word2cate_feat[word] /= word_cnt[word]

save_pickle(word2cate_feat, f'{SAVE_PATH}/word2cate_feat.pkl')
"""