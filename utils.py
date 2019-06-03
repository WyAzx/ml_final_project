import json
import os
import pickle
import numpy as np
from collections import namedtuple

Config = namedtuple('Config', ['BERT_DIR', 'DATA_DIR'])
BertConfig = namedtuple('BertConfig', ['config', 'check_point', 'vocab'])
identity_columns = [
    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
    'muslim', 'black', 'white', 'psychiatric_or_mental_illness'
]


def get_config() -> Config:
    config = Config(**json.load(open('config.json')))
    return config


def get_bert_config(config: Config) -> BertConfig:
    bert_dir = config.BERT_DIR
    bert_config = BertConfig(**{
        'config': os.path.join(bert_dir, 'bert_config.json'),
        'check_point': os.path.join(bert_dir, 'bert_model.ckpt'),
        'vocab': os.path.join(bert_dir, 'vocab.txt')
    })
    return bert_config


def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')


def load_embeddings(path):
    with open(path, 'rb') as f:
        emb_arr = pickle.load(f)
    return emb_arr


def build_matrix(word_index, path):
    embedding_index = load_embeddings(path)
    embedding_matrix = np.zeros((len(word_index) + 1, 300))
    unknown_words = []

    for word, i in word_index.items():
        if i <= len(word_index):
            try:
                embedding_matrix[i] = embedding_index[word]
            except KeyError:
                try:
                    embedding_matrix[i] = embedding_index[word.lower()]
                except KeyError:
                    try:
                        embedding_matrix[i] = embedding_index[word.title()]
                    except KeyError:
                        unknown_words.append(word)
    return embedding_matrix, unknown_words


def get_weights(df):
    # Overall
    weights = np.ones((len(df),)) / 4

    # Subgroup
    weights += (df[identity_columns].fillna(0).values >= 0.5).sum(axis=1).astype(bool).astype(np.int) / 4

    # Background Positive, Subgroup Negative
    weights += (((df['target'].values >= 0.5).astype(bool).astype(np.int) +
                 (df[identity_columns].fillna(0).values < 0.5).sum(axis=1).astype(bool).astype(np.int)) > 1).astype(
        bool).astype(np.int) / 4

    # Background Negative, Subgroup Positive
    weights += (((df['target'].values < 0.5).astype(bool).astype(np.int) +
                 (df[identity_columns].fillna(0).values >= 0.5).sum(axis=1).astype(bool).astype(np.int)) > 1).astype(
        bool).astype(np.int) / 4
    weights /= weights.mean()
    return weights
