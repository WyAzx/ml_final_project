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


def build_bpe_matrix(word_index, model_path='/nfs/cold_project/wangyun/jigsaw/bpe_model/en.wiki.bpe.vs200000.model',
                     vocab_path='/nfs/cold_project/wangyun/jigsaw/bpe_model/data/en/en.wiki.bpe.vs200000.d300.w2v.bin'):
    import sentencepiece as spm
    spr = spm.SentencePieceProcessor()
    spr.Load(model_path)
    # tokens = spr.EncodeAsPieces('this is a test')
    from gensim.models import KeyedVectors
    model = KeyedVectors.load_word2vec_format(vocab_path, binary=True)

    oov = []
    embedding_matrix = np.zeros((len(word_index) + 1, 300))
    for word, i in word_index.items():
        tokens = spr.EncodeAsPieces(word.lower())
        tokens_list = []
        for tok in tokens:
            try:
                tokens_list.append(model[tok])
            except KeyError:
                oov.append(tok)
        bpe = np.average(np.array(tokens_list), axis=0)
        embedding_matrix[i] = bpe
    print(oov)
    return embedding_matrix


def get_weights(df):
    # Overall
    weights = np.ones((len(df),)) / 4

    # Subgroup
    weights += (df[identity_columns].fillna(0).values >= 0.5).sum(axis=1).astype(bool).astype(np.int) / 4

    # Background Positive, Subgroup Negative
    # Background Positive ++
    weights += (((df['target'].values >= 0.5).astype(bool).astype(np.int) +
                 (df[identity_columns].fillna(0).values < 0.5).sum(axis=1).astype(bool).astype(np.int)) > 1).astype(
        bool).astype(np.int) / 4

    # Background Negative, Subgroup Positive
    # Subgroup Negative ++
    weights += (((df['target'].values < 0.5).astype(bool).astype(np.int) +
                 (df[identity_columns].fillna(0).values >= 0.5).sum(axis=1).astype(bool).astype(np.int)) > 1).astype(
        bool).astype(np.int) / 4
    weights /= weights.mean()
    return weights


def get_weights_new(df):
    # Overall
    weights = np.ones((len(df),)) / 4

    # Subgroup
    weights += (df[identity_columns].fillna(0).values >= 0.5).sum(axis=1).astype(bool).astype(np.int) / 4

    # Background Positive, Subgroup Negative
    # Background Positive ++
    weights += (((df['target'].values >= 0.5).astype(bool).astype(np.int) +
                 (df[identity_columns].fillna(0).values < 0.5).sum(axis=1).astype(bool).astype(np.int)) > 1).astype(
        bool).astype(np.int) / 4

    # Background Negative, Subgroup Positive
    # Subgroup Negative ++
    weights += (((df['target'].values < 0.5).astype(bool).astype(np.int) +
                 (df[identity_columns].fillna(0).values >= 0.5).sum(axis=1).astype(bool).astype(np.int)) > 1).astype(
        bool).astype(np.int) * 3 / 4

    weights += (((df['target'].values == 0.0).astype(bool).astype(np.int) +
                 (df[identity_columns].fillna(0).values >= 0.5).sum(axis=1).astype(bool).astype(np.int)) > 1).astype(
        bool).astype(np.int) * 3 / 4
    return weights


def get_weights_new_array(idts, labels):
    weights = np.ones((len(idts),)) / 4

    weights += (idts >= 0.5).sum(axis=1).astype(bool).astype(np.int) / 4

    # Background Positive, Subgroup Negative
    # Background Positive ++
    weights += (((labels >= 0.5).astype(bool).astype(np.int) +
                 (idts < 0.5).sum(axis=1).astype(bool).astype(np.int)) > 1).astype(
        bool).astype(np.int) / 4

    # Background Negative, Subgroup Positive
    # Subgroup Negative ++
    weights += (((labels < 0.5).astype(bool).astype(np.int) +
                 (idts >= 0.5).sum(axis=1).astype(bool).astype(np.int)) > 1).astype(
        bool).astype(np.int) * 3 / 4
    return weights


def get_weights2(df):
    # Overall
    weights = np.ones((len(df),))

    weights += (df[identity_columns].fillna(0).values >= 0.5).sum(axis=1).astype(bool).astype(np.int)

    weights += (((df['target'].values < 0.5).astype(bool).astype(np.int) +
                 (df[identity_columns].fillna(0).values >= 0.5).sum(axis=1).astype(bool).astype(np.int)) > 1).astype(
        bool).astype(np.int)

    weights += (((df['target'].values < 0.5).astype(bool).astype(np.int) +
                 (df['homosexual_gay_or_lesbian'].fillna(0).values >= 0.5).astype(np.int)) > 1).astype(np.int) * 10
    weights += (((df['target'].values < 0.5).astype(bool).astype(np.int) +
                 (df['white'].fillna(0).values >= 0.5).astype(np.int)) > 1).astype(np.int) * 10

    weights += (((df['target'].values >= 0.5).astype(bool).astype(np.int) +
                 (df['homosexual_gay_or_lesbian'].fillna(0).values >= 0.5).astype(np.int)) > 1).astype(np.int) * 5
    weights += (((df['target'].values >= 0.5).astype(bool).astype(np.int) +
                 (df['white'].fillna(0).values >= 0.5).astype(np.int)) > 1).astype(np.int) * 5

    return weights


def get_weight_ng(df):
    # Overall
    weights = np.ones((len(df),))

    weights -= (df[identity_columns].fillna(0).values >= 0.5).sum(axis=1).astype(bool).astype(np.int) / 3

    weights -= (((df['target'].values >= 0.5).astype(bool).astype(np.int) +
                 (df[identity_columns].fillna(0).values >= 0.5).sum(axis=1).astype(bool).astype(np.int)) > 1).astype(
        bool).astype(np.int) / 5

    return weights


def get_weight_2(df):
    for column in identity_columns + ['target']:
        df[column] = np.where(df[column] >= 0.5, True, False)
    sample_weights = np.ones(len(df), dtype=np.float32)
    sample_weights += df[identity_columns].sum(axis=1)
    sample_weights += df['target'] * (~df[identity_columns]).sum(axis=1)
    sample_weights += (~df['target']) * df[identity_columns].sum(axis=1) * 5
    sample_weights /= sample_weights.mean()
    return sample_weights.values


if __name__ == '__main__':
    import pickle
    word_index = pickle.load(open('new_processed_data/word_index.pkl', 'rb'))

    bpe_embedding_matrix = build_bpe_matrix(word_index)

    pickle.dump(bpe_embedding_matrix, open('new_processed_data/bpe_embedding_matrix.pkl', 'wb'))