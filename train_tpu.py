import os
import sys
sys.path.append('keras_layers')
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from data_loader import tokenize_examples, seq_padding, DataGenerator, AllDataGenerator, GeneralDataGenerator
from keras_bert.optimizers import AdamWarmup
from loggers import Logger
from models.bert_base_model import get_bert_base_model, get_bert_multi_model, get_bert_multi_layers_model
from utils import get_bert_config, get_config, get_weights_new, BertConfig
from bert import tokenization
from keras_bert import get_custom_objects


IDENTITY_COLUMNS = [
    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
    'muslim', 'black', 'white', 'psychiatric_or_mental_illness'
]
AUX_COLUMNS = ['target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']

model_folder = 'data/uncased_L-12_H-768_A-12'
bert_config = BertConfig(**{
  'config': os.path.join(model_folder, 'bert_config.json'),
  'check_point': os.path.join(model_folder, 'bert_model.ckpt'),
  'vocab': os.path.join(model_folder, 'vocab.txt')
})
vocab_file = os.path.join(model_folder, 'vocab.txt')
tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=True)
#
# tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
#         FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)


def convert_lines(example, max_seq_length, tokenizer, prunc='pre'):
  max_seq_length -= 2
  all_tokens = []
  longer = 0
  for i in range(example.shape[0]):
    tokens_a = tokenizer.tokenize(example[i])
    if len(tokens_a) > max_seq_length:
      if prunc == 'pre':
        tokens_a = tokens_a[:max_seq_length]
      elif prunc == 'post':
        tokens_a = tokens_a[-max_seq_length:]
      else:
        tokens_a = tokens_a[:(max_seq_length // 2)] + tokens_a[-(max_seq_length // 2):]
      longer += 1
    one_token = tokenizer.convert_tokens_to_ids(["[CLS]"] + tokens_a + ["[SEP]"])
    all_tokens.append(one_token)
  print(longer)
  return np.array(all_tokens)


def train_bert_on_tpu():
  df = pd.read_csv('data/train.csv')
  weight_df = pd.read_csv('data/train_weight.csv')
  texts = df['comment_text'].values
  # texts = convert_lines(texts, 512, tokenizer, prunc='ei')

  label = df['target'].values
  aux_label = df[AUX_COLUMNS].values

  weight = weight_df['weight'].values

  del df
  del weight_df

  train_text, _, train_label, _, train_aux, _, train_weights, _ = train_test_split(texts, label, aux_label, weight,
                                                                                   test_size=0.055, random_state=59)

  def pad_fn(ts):
    ts = convert_lines(np.array(ts), 512, tokenizer, prunc='ei')
    return seq_padding(ts, truncate=False)
  lw = 1 / np.mean(train_weights)
  train_gen = GeneralDataGenerator(inputs=[train_text], outputs=[train_label, train_aux],
                                   sample_weights=[train_weights, np.ones_like(train_weights)], batch_size=64,
                                   pad_fn=[pad_fn])

  model = get_bert_multi_model(bert_config)

  lr = 2e-5
  weight_decay = 0.01
  decay_steps = 1 * len(train_gen)
  warmup_steps = int(0.1 * decay_steps)

  optimizer = AdamWarmup(
    decay_steps=decay_steps,
    warmup_steps=warmup_steps,
    lr=lr,
    weight_decay=weight_decay,
  )

  strategy = tf.contrib.tpu.TPUDistributionStrategy(
    tf.contrib.cluster_resolver.TPUClusterResolver("node-2", zone="us-central1-b", project='studied-acronym-235702')
  )

  with tf.keras.utils.custom_object_scope(get_custom_objects()):
    tpu_model = tf.contrib.tpu.keras_to_tpu_model(model, strategy=strategy)
    tpu_model.compile(loss='binary_crossentropy', optimizer=optimizer, loss_weights=[lw, 1.])

    tpu_model.fit_generator(train_gen.__iter__(),
                                 steps_per_epoch=len(train_gen),
                                 epochs=1,
                                 max_queue_size=100,
                                 )
    model.save('save_models/bert.weights-ei.h5')


if __name__ == '__main__':
    train_bert_on_tpu()