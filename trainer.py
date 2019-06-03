import pandas as pd
import numpy as np
import gc
import os
from keras.models import Model
from nltk.tokenize.treebank import TreebankWordTokenizer

tk = TreebankWordTokenizer()
from keras.preprocessing import text, sequence
from sklearn.model_selection import train_test_split
from data_loader import GeneralDataGenerator, GeneralPredictGenerator, SeqDataGenerator
from loggers import Logger
from utils import build_matrix, get_weights
from models.lstm_model import get_lstm_model
from tqdm import tqdm
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))
GEN = 1

IDENTITY_COLUMNS = [
    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
    'muslim', 'black', 'white', 'psychiatric_or_mental_illness'
]
AUX_COLUMNS = ['target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']
TEXT_COLUMN = 'comment_text'
TARGET_COLUMN = 'target'


class Trainer(object):

    def __init__(self, model: Model, save_model: Model, model_name: str):
        self.model = model
        self.save_model = save_model
        self.model_name = model_name

    def train(self, train_gen, logger, epochs):
        self.model.fit_generator(
            train_gen, len(train_gen), epochs=epochs, callbacks=[logger]
        )

    def save(self, save_path):
        self.model.save(save_path)


def train():
    ################
    #   LOAD DATA  #
    ################
    # df = pd.read_csv('processed_data/train_tok.csv')
    # # tqdm.pandas()
    # # texts = df['comment_text'].fillna('none').progress_apply(lambda s: ' '.join(tk.tokenize(s)))
    # # df['comment_text'] = texts
    # # df.to_csv('processed_data/train_tok.csv', index=False)
    # texts = df['comment_text'].values
    # labels_aux = df[AUX_COLUMNS].values
    # labels = df[TARGET_COLUMN].values
    # weights = get_weights(df)
    #
    # tokenizer = text.Tokenizer(filters='', lower=False)
    # tokenizer.fit_on_texts(list(texts))
    # texts = tokenizer.texts_to_sequences(texts)
    #
    # df['token_text'] = texts
    #
    # crawl_matrix, unknown_words_crawl = build_matrix(tokenizer.word_index, 'embedding/crawl-300d-2M.pkl')
    # print('n unknown words (crawl): ', len(unknown_words_crawl))
    #
    # glove_matrix, unknown_words_glove = build_matrix(tokenizer.word_index, 'embedding/glove.840B.300d.pkl')
    # print('n unknown words (glove): ', len(unknown_words_glove))
    #
    # max_features = len(tokenizer.word_index) + 1
    # print('Vocab Size:', max_features)
    #
    # embedding_matrix = np.concatenate([crawl_matrix, glove_matrix], axis=-1)
    # print('Embedding shape:', embedding_matrix.shape)
    # del crawl_matrix
    # del glove_matrix
    # gc.collect()
    # #
    batch_size = 512
    #
    # train_texts, val_texts, train_df, val_df, train_indxs, val_indxs = train_test_split(texts, df, range(len(df)),
    #                                                                                     random_state=59,
    #                                                                                     test_size=0.055)
    # train_label, train_aux_label, train_weight = labels[train_indxs], labels_aux[train_indxs], weights[train_indxs]
    #
    # val_df = val_df.reset_index()
    import pickle
    # pickle.dump(embedding_matrix, open('processed_data/emb.pkl', 'wb'))
    # pickle.dump(train_texts, open('processed_data/train_texts.pkl', 'wb'))
    # pickle.dump(val_texts, open('processed_data/val_texts.pkl', 'wb'))
    # pickle.dump(train_label, open('processed_data/train_label.pkl', 'wb'))
    # pickle.dump(train_aux_label, open('processed_data/train_aux_label.pkl', 'wb'))
    # pickle.dump(train_weight, open('processed_data/train_weight.pkl', 'wb'))
    # val_df.to_csv('processed_data/val.csv', index=False)

    embedding_matrix = pickle.load(open('processed_data/emb.pkl', 'rb'))
    train_texts = pickle.load(open('processed_data/train_texts.pkl', 'rb'))
    val_texts = pickle.load(open('processed_data/val_texts.pkl', 'rb'))
    train_label = pickle.load(open('processed_data/train_label.pkl', 'rb'))
    train_aux_label = pickle.load(open('processed_data/train_aux_label.pkl', 'rb'))
    train_weight = pickle.load(open('processed_data/train_weight.pkl', 'rb'))
    val_df = pd.read_csv('processed_data/val.csv')

    if GEN != 0:
        train_gen = GeneralDataGenerator(inputs=[train_texts], outputs=[train_label, train_aux_label],
                                         sample_weights=[train_weight, np.ones_like(train_weight)], batch_size=batch_size)
    else:
        train_gen = SeqDataGenerator(train_texts, train_label, train_aux_label, train_weight)
    val_gen = GeneralPredictGenerator(text=val_df['token_text'].values)

    model, save_model = get_lstm_model(embedding_matrix, len(AUX_COLUMNS))

    logger = Logger(save_model, 'lstm_baseline', val_gen, val_df)

    model.fit_generator(train_gen.__iter__(), len(train_gen), epochs=5, callbacks=[logger])


if __name__ == '__main__':
    train()
