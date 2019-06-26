import os
import sys

from evaluator import JigsawEvaluator

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
sys.path.append('keras_layers')
from models.gpt_models import get_gpt_model
from utils import get_weights_new, get_weights2

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.utils import multi_gpu_model
from loggers import Logger
from sklearn.model_selection import train_test_split
from keras_bert.optimizers import AdamWarmup
from data_loader import tokenize_examples, seq_padding, DataGenerator, AllDataGenerator, GeneralDataGenerator, \
    GeneralPredictGenerator
from keras.backend.tensorflow_backend import set_session

from keras_gpt_2 import load_trained_model_from_checkpoint, get_bpe_from_files, generate

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

TARGET_COLUMN = 'target'
IDENTITY_COLUMNS = [
    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
    'muslim', 'black', 'white', 'psychiatric_or_mental_illness'
]
AUX_COLUMNS = ['target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']

model_folder = 'gpt_models/117M'
config_path = os.path.join(model_folder, 'hparams.json')
checkpoint_path = os.path.join(model_folder, 'model.ckpt')
encoder_path = os.path.join(model_folder, 'encoder.json')
vocab_path = os.path.join(model_folder, 'vocab.bpe')


def train_gpt():
    bpe = get_bpe_from_files(encoder_path, vocab_path)
    import pickle
    # with open('tok_text_uncased.pkl', 'rb') as h:
    #     text = pickle.load(h)
    with open('y_train.pkl', 'rb') as h:
        label = pickle.load(h)
    with open('y_aux.pkl', 'rb') as h:
        aux = pickle.load(h)
    iden_df = pd.read_csv('processed_data/train_tok_iden.csv')
    weights = get_weights_new(iden_df)
    del iden_df
    df = pd.read_csv('new_processed_data/train.csv')
    text = df['comment_text'].values
    del df

    train_text, _, train_label, _, train_aux, _, train_weights, _ = train_test_split(text, label, aux, weights,
                                                                                     test_size=0.055, random_state=59)

    def pad_fn(ts):
        ts = [bpe.encode(t)[:512] for t in ts]
        return seq_padding(ts, truncate=False)
    train_gen = GeneralDataGenerator(inputs=[train_text, ], outputs=[train_label, train_aux],
                                     sample_weights=[train_weights, np.ones_like(train_weights)], batch_size=16,
                                     pad_fn=[pad_fn, ])

    with tf.device('/cpu:0'):
        model = get_gpt_model(config_path, checkpoint_path)

    lr = 2e-5
    weight_decay = 0.01
    bsz = 32
    decay_steps = 2 * len(train_gen)
    warmup_steps = int(0.05 * decay_steps)

    optimizer = AdamWarmup(
        decay_steps=decay_steps,
        warmup_steps=warmup_steps,
        lr=lr,
        weight_decay=weight_decay,
    )
    lw = 1 / np.mean(train_weights)
    model.load_weights('save_models/gpt.weights-new_weight.h5')
    parallel_model = multi_gpu_model(model, gpus=2)
    parallel_model.compile(loss='binary_crossentropy', optimizer=optimizer, loss_weights=[lw, 1.])
    parallel_model.fit_generator(train_gen.__iter__(),
                                 steps_per_epoch=len(train_gen),
                                 epochs=2,
                                 max_queue_size=100,
                                 initial_epoch=1
                                 )
    model.save('save_models/gpt.weights-new_weight-2.h5')
    print("DONE")


def val_gpt():
    bpe = get_bpe_from_files(encoder_path, vocab_path)
    df = pd.read_csv('new_processed_data/train.csv')
    text = df['comment_text'].fillna('none').values
    idens = df[IDENTITY_COLUMNS].values
    label = df['target'].values
    _, test_text, _, test_label, _, test_iden = train_test_split(text, label, idens, test_size=0.055, random_state=59)
    model = get_gpt_model(config_path, checkpoint_path)
    def pad_fn(ts):
        ts = [bpe.encode(t)[:512] for t in ts]
        return seq_padding(ts, truncate=False)

    print(bpe.encode(test_text[0]))
    model.load_weights('save_models/gpt.weights-new_weight-2.h5')

    val_gen = GeneralPredictGenerator(test_text, 24, pad_fn=pad_fn)

    res = model.predict_generator(val_gen.__iter__(), len(val_gen))[0].flatten()

    eva = JigsawEvaluator(test_label, test_iden)
    final_auc, overall_auc, sub_auc, bpsn_auc, bnsp_auc, bias_metrics = eva.get_final_metric(res)
    print('Final AUC:{}\nOverall AUC:{}\nSub AUC:{}\nBPSN AUC:{}\nBNSP AUC:{}\n'.format(final_auc, overall_auc,
                                                                                        sub_auc, bpsn_auc,
                                                                                        bnsp_auc))
    print('Detail Bias:\n', bias_metrics)

    res.save('results/gpt_res_2.npy')


def get_weight():
    iden_df = pd.read_csv('processed_data/train_tok_iden.csv')
    weights = get_weights_new(iden_df)
    w0 = get_weights2(iden_df)
    del iden_df
    df = pd.read_csv('new_processed_data/train.csv')
    df['weight'] = weights
    df['weight0'] = w0

    df.to_csv('new_processed_data/train_weight.csv', index=False)


if __name__ == '__main__':
    # train_gpt()
    # get_weight()
    val_gpt()