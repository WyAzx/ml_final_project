import os
import sys
sys.path.append('keras_layers')
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.utils import multi_gpu_model
from sklearn.model_selection import train_test_split
from data_loader import tokenize_examples, seq_padding, DataGenerator
from bert.tokenization import FullTokenizer
from keras_bert.optimizers import AdamWarmup
from loggers import Logger
from models.bert_base_model import get_bert_base_model
from utils import get_bert_config, get_config
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))


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
    decay_steps = 4 * len(train_text) // bsz
    warmup_steps = int(0.1 * decay_steps)

    optimizer = AdamWarmup(
        decay_steps=decay_steps,
        warmup_steps=warmup_steps,
        lr=lr,
        kernel_weight_decay=weight_decay,
    )

    parallel_model = multi_gpu_model(model, gpus=4)
    parallel_model.compile(loss='binary_crossentropy', optimizer=optimizer)
    parallel_model.fit_generator(train_gen.__iter__(),
                                 steps_per_epoch=len(train_gen),
                                 epochs=2,
                                 callbacks=[logger],
                                 max_queue_size=100
                                 )


def load_data(data_path):
    print(data_path)
    data = pd.read_csv(data_path)
    text, label = data['comment_text'].values, data.target.values
    return text, label


if __name__ == '__main__':
    train_on_train_test_split()
