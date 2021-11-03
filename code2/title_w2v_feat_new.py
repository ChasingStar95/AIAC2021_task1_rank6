import os
import jieba
import gensim
import itertools
import numpy as np
import pandas as pd
from collections import Counter
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from multiprocessing import Pool

from utils import save_pickle, LoadDataset

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)

DEBUG = False

DATA_PATH = '../data/new_data/'
SAVE_PATH = f'{DATA_PATH}/title_w2v_final'
if not os.path.exists(SAVE_PATH):
    os.mkdir(SAVE_PATH)

if DEBUG:
    SAVE_PATH = f'{SAVE_PATH}/debug'
    if not os.path.exists(SAVE_PATH):
        os.mkdir(SAVE_PATH)

EMBEDDING_DIM = 300
MIN_CNT = 3
WINDOW_SIZE = 30

def parallelize_df_func(df, func, num_partitions=16, n_jobs=8):
    df_split = np.array_split(df, num_partitions)
    pool = Pool(n_jobs)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df

def jieba_seg(df):
    return df.apply(lambda x: jieba.lcut(x))

def chose_seg(df):
    return df.apply(lambda x: [word for word in x if word in chosen_word])

dataset = LoadDataset(data_path=DATA_PATH)

whole_df = dataset.df
# whole_df['title_seg'] = whole_df['title'].apply(lambda x: jieba.lcut(x))
whole_df['title_seg'] = parallelize_df_func(whole_df['title'], jieba_seg)

word_cnt = Counter(list(itertools.chain(*whole_df.title_seg.values.tolist())))
word_cnt_df = pd.DataFrame()
word_cnt_df['word'] = word_cnt.keys()
word_cnt_df['cnt'] = word_cnt.values()
word_cnt_df = word_cnt_df.sort_values('cnt', ascending=False).reset_index(drop=True)


chosen_word_list = word_cnt_df.loc[word_cnt_df.cnt >= MIN_CNT].word.values.tolist()
chosen_word = {}
for _idx, _word in enumerate(chosen_word_list):
    chosen_word[_word] = _idx
whole_df['title_seg_chosen'] = parallelize_df_func(whole_df['title_seg'], chose_seg)

# for NEG in [3, 5, 7, 9]:
#     print('NEG: ', NEG)
model = Word2Vec(whole_df.title_seg_chosen.values.tolist(),
                     vector_size=EMBEDDING_DIM, window=30, epochs=8, workers=20, seed=2020, min_count=1,
                     sg=1, negative=5,
                     )

word2emb = {}
for word in chosen_word:
    word2emb[word] = model.wv[word]
save_pickle(word2emb, f'{SAVE_PATH}/w2v_emb_new.pkl')

# ##### ASR
# print('asr word start.')
# whole_df['asr_seg'] = parallelize_df_func(whole_df['asr_text'].astype(str), jieba_seg)
#
# word_cnt = Counter(list(itertools.chain(*whole_df.asr_seg.values.tolist())))
# word_cnt_df = pd.DataFrame()
# word_cnt_df['word'] = word_cnt.keys()
# word_cnt_df['cnt'] = word_cnt.values()
# word_cnt_df = word_cnt_df.sort_values('cnt', ascending=False).reset_index(drop=True)
#
#
# chosen_word_list = word_cnt_df.loc[word_cnt_df.cnt >= MIN_CNT].word.values.tolist()
# chosen_word = {}
# for _idx, _word in enumerate(chosen_word_list):
#     chosen_word[_word] = _idx
# whole_df['asr_seg_chosen'] = parallelize_df_func(whole_df['asr_seg'], chose_seg)
#
# # for NEG in [3, 5, 7, 9]:
# #     print('NEG: ', NEG)
# model = Word2Vec(whole_df.asr_seg_chosen.values.tolist(),
#                      vector_size=EMBEDDING_DIM, window=10, epochs=8, workers=20, seed=2020, min_count=1,
#                      sg=1, negative=5,
#                      )
#
# word2emb = {}
# for word in chosen_word:
#     word2emb[word] = model.wv[word]
# save_pickle(word2emb, f'{SAVE_PATH}/asr_w2v_emb_new.pkl')
#
#
# print('asr char start.')
# whole_df['asr_seg'] = whole_df['asr_text'].apply(lambda x: list(str(x)))
#
# word_cnt = Counter(list(itertools.chain(*whole_df.asr_seg.values.tolist())))
# word_cnt_df = pd.DataFrame()
# word_cnt_df['word'] = word_cnt.keys()
# word_cnt_df['cnt'] = word_cnt.values()
# word_cnt_df = word_cnt_df.sort_values('cnt', ascending=False).reset_index(drop=True)
#
#
# chosen_word_list = word_cnt_df.loc[word_cnt_df.cnt >= MIN_CNT].word.values.tolist()
# chosen_word = {}
# for _idx, _word in enumerate(chosen_word_list):
#     chosen_word[_word] = _idx
# whole_df['asr_seg_chosen'] = parallelize_df_func(whole_df['asr_seg'], chose_seg)
#
# # for NEG in [3, 5, 7, 9]:
# #     print('NEG: ', NEG)
# model = Word2Vec(whole_df.asr_seg_chosen.values.tolist(),
#                      vector_size=EMBEDDING_DIM, window=10, epochs=8, workers=20, seed=2020, min_count=1,
#                      sg=1, negative=5,
#                      )
#
# word2emb = {}
# for word in chosen_word:
#     word2emb[word] = model.wv[word]
# save_pickle(word2emb, f'{SAVE_PATH}/asr_char_w2v_emb_new.pkl')
