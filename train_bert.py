import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
sys.path.append('keras_layers')
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.utils import multi_gpu_model
from sklearn.model_selection import train_test_split
from data_loader import tokenize_examples, seq_padding, DataGenerator, AllDataGenerator, GeneralDataGenerator
from bert.tokenization import FullTokenizer
from keras_bert.optimizers import AdamWarmup
from loggers import Logger
from models.bert_base_model import get_bert_base_model, get_bert_multi_model, get_bert_multi_layers_model
from utils import get_bert_config, get_config, get_weights_new
from keras.backend.tensorflow_backend import set_session
from keras.optimizers import Adam
from focal_loss import focal_loss
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

TARGET_COLUMN = 'target'
IDENTITY_COLUMNS = [
    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
    'muslim', 'black', 'white', 'psychiatric_or_mental_illness'
]
AUX_COLUMNS = ['target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']


def train_ml_all_set():
    train_config = get_config()
    bert_config = get_bert_config(train_config)
    import pickle
    with open('tok_text_uncased.pkl', 'rb') as h:
        text = pickle.load(h)
    with open('y_train.pkl', 'rb') as h:
        label = pickle.load(h)
    with open('y_aux.pkl', 'rb') as h:
        aux = pickle.load(h)
    iden_df = pd.read_csv('processed_data/train_tok_iden.csv')
    weights = get_weights_new(iden_df)
    del iden_df

    train_text, _, train_label, _, train_aux, _, train_weights, _ = train_test_split(text, label, aux, weights,
                                                                                     test_size=0.055, random_state=59)
    train_seg = [[0 for _ in t] for t in train_text]

    train_gen = GeneralDataGenerator(inputs=[train_text, train_seg], outputs=[train_label, train_aux],
                                     sample_weights=[train_weights, np.ones_like(train_weights)],
                                     pad_fn=[lambda x: seq_padding(x, truncate=False),
                                             lambda x: seq_padding(x, truncate=False)], batch_size=64)

    with tf.device('/cpu:0'):
        model = get_bert_multi_model(bert_config)

    # optimizer = Adam(lr=2e-5)
    # OPTIMIZER PARAMs
    lr = 2e-5
    weight_decay = 0.01
    bsz = 32
    decay_steps = 1 * len(train_gen)
    warmup_steps = int(0.1 * decay_steps)

    optimizer = AdamWarmup(
        decay_steps=decay_steps,
        warmup_steps=warmup_steps,
        lr=lr,
        weight_decay=weight_decay,
        weight_decay_pattern=['embeddings', 'kernel', 'W1', 'W2', 'Wk', 'Wq', 'Wv', 'Wo'],
    )

    parallel_model = multi_gpu_model(model, gpus=4)
    # parallel_model.compile(loss=[focal_loss(gamma=2., alpha=.25), 'binary_crossentropy'], optimizer=optimizer)
    parallel_model.compile(loss='binary_crossentropy', optimizer=optimizer)

    parallel_model.fit_generator(train_gen.__iter__(),
                                 steps_per_epoch=len(train_gen),
                                 epochs=1,
                                 max_queue_size=20,
                                 )
    model.save('save_models/bert.weights-large-nw.h5')
    # print('SAVED')
    # parallel_model.fit_generator(train_gen.__iter__(),
    #                              steps_per_epoch=len(train_gen),
    #                              epochs=1,
    #                              max_queue_size=20,
    #                              )
    # model.save('save_models/bert.weights-uncased-ml2-e2.h5')
    print("DONE")


def train_on_all_set():
    train_config = get_config()
    bert_config = get_bert_config(train_config)
    import pickle
    with open('tok_text_uncased.pkl', 'rb') as h:
        text = pickle.load(h)
    with open('y_train.pkl', 'rb') as h:
        label = pickle.load(h)
    with open('y_aux.pkl', 'rb') as h:
        aux = pickle.load(h)
    iden_df = pd.read_csv('processed_data/train_tok_iden.csv')
    weights = get_weights_new(iden_df)
    del iden_df
    lw = 1 / np.mean(weights)
    train_seg = [[0 for _ in t] for t in text]

    train_gen = GeneralDataGenerator(inputs=[text, train_seg], outputs=[label, aux],
                                     sample_weights=[weights, np.ones_like(weights)], batch_size=32,
                                     pad_fn=[lambda x: seq_padding(x, truncate=False),
                                             lambda x: seq_padding(x, truncate=False)])
    # train_gen = AllDataGenerator(text, label, aux, sample_weight)

    with tf.device('/cpu:0'):
        model = get_bert_multi_model(bert_config)
    # model.load_weights('save_models/bert.weights.h5')

    # OPTIMIZER PARAMs
    lr = 2e-5
    weight_decay = 0.01
    bsz = 32
    decay_steps = 1 * len(train_gen)
    warmup_steps = int(0.1 * decay_steps)

    optimizer = AdamWarmup(
        decay_steps=decay_steps,
        warmup_steps=warmup_steps,
        lr=lr,
        weight_decay=weight_decay,
    )

    parallel_model = multi_gpu_model(model, gpus=2)
    parallel_model.compile(loss='binary_crossentropy', optimizer=optimizer, loss_weights=[lw, 1.])
    parallel_model.fit_generator(train_gen.__iter__(),
                                 steps_per_epoch=len(train_gen),
                                 epochs=1,
                                 max_queue_size=100,
                                 )
    model.save('save_models/bert.weights-uncased-new_weight_all.h5')
    print("DONE")


def train_on_train_test_split():
    train_config = get_config()
    bert_config = get_bert_config(train_config)
    cased = train_config.BERT_DIR.split('/')[-1].startswith('cased')
    tokenizer = FullTokenizer(bert_config.vocab, do_lower_case=cased)

    with tf.device('/cpu:0'):
        model = get_bert_base_model(bert_config)

    text, label = load_data(os.path.join(train_config.DATA_DIR, 'train.csv'))
    train_text, val_text, train_label, val_label = train_test_split(text, label, test_size=0.055, random_state=59)
    train_gen = DataGenerator(train_text, train_label, tokenizer, batch_size=32)

    val_text = tokenize_examples(val_text, tokenizer, max_len=512)
    val_text = seq_padding(val_text)

    logger = Logger(model=model, val_text=val_text, val_label=(val_label > 0.5).astype(np.float32))

    # OPTIMIZER PARAMs
    lr = 2e-5
    weight_decay = 0.01
    bsz = 32
    decay_steps = 1 * len(train_gen)
    warmup_steps = int(0.1 * decay_steps)

    optimizer = AdamWarmup(
        decay_steps=decay_steps,
        warmup_steps=warmup_steps,
        lr=lr,
        weight_decay=weight_decay,
    )

    parallel_model = multi_gpu_model(model, gpus=4)
    parallel_model.compile(loss='binary_crossentropy', optimizer=optimizer)
    parallel_model.fit_generator(train_gen.__iter__(),
                                 steps_per_epoch=len(train_gen),
                                 epochs=1,
                                 callbacks=[logger],
                                 max_queue_size=100
                                 )


def load_data(data_path):
    print(data_path)
    data = pd.read_csv(data_path)
    text, label = data['comment_text'].values, data.target.values
    return text, label


def tokenize_bert():
    train_config = get_config()
    bert_config = get_bert_config(train_config)
    uncased = train_config.BERT_DIR.split('/')[-1].startswith('uncased')
    tokenizer = FullTokenizer(bert_config.vocab, do_lower_case=uncased)
    text, _ = load_data(os.path.join(train_config.DATA_DIR, 'train.csv'))
    tok_text = tokenize_examples(text, tokenizer, max_len=512)
    import pickle
    pickle.dump(tok_text, open('tok_text_uncased.pkl', 'wb'))


def get_weight():
    train_config = get_config()
    train_df = pd.read_csv(os.path.join(train_config.DATA_DIR, 'train.csv'))
    y_train = train_df[TARGET_COLUMN].values
    y_aux_train = train_df[AUX_COLUMNS].values
    for column in IDENTITY_COLUMNS + [TARGET_COLUMN]:
        train_df[column] = np.where(train_df[column] >= 0.5, True, False)
    sample_weights = np.ones(len(y_train), dtype=np.float32)
    sample_weights += train_df[IDENTITY_COLUMNS].sum(axis=1)
    sample_weights += train_df[TARGET_COLUMN] * (~train_df[IDENTITY_COLUMNS]).sum(axis=1)
    sample_weights += (~train_df[TARGET_COLUMN]) * train_df[IDENTITY_COLUMNS].sum(axis=1) * 5
    sample_weights /= sample_weights.mean()
    import pickle
    pickle.dump(y_aux_train, open('y_aux.pkl', 'wb'))
    pickle.dump(sample_weights, open('sample_weight.pkl', 'wb'))
    pickle.dump(y_train, open('y_train.pkl', 'wb'))


if __name__ == '__main__':
    # train_on_train_test_split()
    # tokenize_bert()
    # get_weight()
    # train_on_all_set()
    train_ml_all_set()
