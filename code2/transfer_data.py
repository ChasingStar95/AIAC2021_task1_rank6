import json
import functools
import io
import os
import struct
import typing
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

DEBUG = False
SAVE_PATH = '../data/new_data'
if not os.path.exists(SAVE_PATH):
    os.mkdir(SAVE_PATH)
if DEBUG:
    SAVE_PATH = f'{SAVE_PATH}/debug'
    if not os.path.exists(SAVE_PATH):
        os.mkdir(SAVE_PATH)

MATRIX_PATH = f'{SAVE_PATH}/video'
if not os.path.exists(MATRIX_PATH):
    os.mkdir(MATRIX_PATH)

def _parse_record_tr(example_photo):
    features = {'id': tf.io.FixedLenFeature([], tf.string),
                'tag_id': tf.io.VarLenFeature(tf.int64),
                'category_id': tf.io.VarLenFeature(tf.int64),
                'title': tf.io.FixedLenFeature([], tf.string),
                'frame_feature': tf.io.VarLenFeature(tf.string),
                'asr_text': tf.io.FixedLenFeature([], tf.string),
                }
    parsed_features = tf.io.parse_single_example(example_photo, features=features)
    return parsed_features
def _parse_record_te(example_photo):
    features = {'id': tf.io.FixedLenFeature([], tf.string),
                # 'tag_id': tf.io.VarLenFeature(tf.int64),
                # 'category_id': tf.io.VarLenFeature(tf.int64),
                'title': tf.io.FixedLenFeature([], tf.string),
                'frame_feature': tf.io.VarLenFeature(tf.string),
                'asr_text': tf.io.FixedLenFeature([], tf.string),
                }
    parsed_features = tf.io.parse_single_example(example_photo, features=features)
    return parsed_features

def _sample(frames, max_frames=32):
    frames = frames.numpy()
    frames_len = len(frames)
    num_frames = min(frames_len, max_frames)
    num_frames = np.array([num_frames], dtype=np.int32)

    average_duration = frames_len // max_frames
    if average_duration == 0:
        return [frames[min(i, frames_len - 1)] for i in range(max_frames)], num_frames
    else:
        offsets = np.multiply(list(range(max_frames)), average_duration) + average_duration // 2
        return [frames[i] for i in offsets], num_frames

def load_data(input_file, is_train=True):
    id_list = []
    tag_list = []
    category_id_list = []
    title_list = []
    asr_text_list = []
    num_frames_list = []

    dataset = tf.data.TFRecordDataset(input_file)
    if is_train:
        dataset = dataset.map(_parse_record_tr)
    else:
        dataset = dataset.map(_parse_record_te)
    for idx, data in enumerate(tqdm(dataset)):
        id = data['id'].numpy().decode(encoding='utf-8')
        if is_train:
            tag_id = tf.sparse.to_dense(data['tag_id']).numpy()# .decode(encoding='utf-8')
            category_id = tf.sparse.to_dense(data['category_id']).numpy()
        title = data['title'].numpy().decode(encoding='utf-8')
        frame_feature = tf.sparse.to_dense(data['frame_feature'])
        frames, num_frames = tf.py_function(_sample, [frame_feature], [tf.string, tf.int32])
        frames_embedding = tf.map_fn(lambda x: tf.io.decode_raw(x, out_type=tf.float16), frames, dtype=tf.float16).numpy()
        asr_text = data['asr_text'].numpy().decode(encoding='utf-8')

        id_list.append(id)
        if is_train:
            tag_list.append(' '.join([str(tag) for tag in tag_id]))
            category_id_list.append(category_id[0])
        title_list.append(str(title))
        asr_text_list.append(str(asr_text))
        num_frames_list.append(num_frames.numpy()[0])
        # print(id, frames_embedding)
        np.save(f'{MATRIX_PATH}/{id}.npy', frames_embedding)
        # print(id, np.load(f'{MATRIX_PATH}/{id}.npy'))
        # print('id: ', id)
        # print('tag_id: ', tag_id, len(tag_id))
        # print('category_id: ', category_id)
        # print('title: ', title)
        # print('frames_embedding: ', frames_embedding, frames_embedding.shape)
        # print('asr_text: ', asr_text)
        if DEBUG:
            if idx > 10:
                break
    df = pd.DataFrame()
    df['id'] = id_list
    if is_train:
        df['tag_id'] = tag_list
        df['category_id'] = category_id_list
    df['title'] = title_list
    df['asr_text'] = asr_text_list
    df['num_frames'] = num_frames_list

    return df

whole_df = []

print('Pairwise:')
pairwise_df = load_data('../data/pairwise/pairwise.tfrecords')
pairwise_df['flag'] = 'pairwise'
whole_df.append(pairwise_df)

print('Test A:')
test_a_df = load_data('../data/test_a/test_a.tfrecords', is_train=False)
test_a_df['flag'] = 'test_a'
whole_df.append(test_a_df)

print('Test B:')
test_b_df = load_data('../data/test_b/test_b.tfrecords', is_train=False)
test_b_df['flag'] = 'test_b'
whole_df.append(test_b_df)

for i in range(20):
    print(f'Pointwise: {i}')
    file_name = f'../data/pointwise/pretrain_{i}.tfrecords'
    pointwise_df = load_data(file_name, is_train=True)
    pointwise_df['flag'] = 'pointwise'
    whole_df.append(pointwise_df)
whole_df = pd.concat(whole_df).reset_index(drop=True)
whole_df.to_csv(f'{SAVE_PATH}/whole_df.csv', index=False, sep='\t', encoding='gb18030')


whole_df = pd.read_csv(f'{SAVE_PATH}/whole_df.csv', sep='\t', encoding='gb18030')

print('Test B:')
test_b_df = load_data('../data/test_b/test_b.tfrecords', is_train=False)
test_b_df['flag'] = 'test_b'
whole_df = pd.concat([whole_df, test_b_df]).reset_index(drop=True)
whole_df.to_csv(f'{SAVE_PATH}/whole_df_new.csv', index=False, sep='\t', encoding='gb18030')
