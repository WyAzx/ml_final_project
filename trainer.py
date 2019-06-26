import pickle

import pandas as pd
import numpy as np
import gc
import os
from keras.models import Model
from keras.optimizers import RMSprop, Adam
from keras.utils import multi_gpu_model
from nltk.tokenize.treebank import TreebankWordTokenizer
import sys

from models.cnn_model import get_dcnn_model
import keras.backend as K

sys.path.append('keras_layers')
from keras_bert import AdamWarmup

tk = TreebankWordTokenizer()
from keras.preprocessing import text, sequence
from sklearn.model_selection import train_test_split, StratifiedKFold
from data_loader import GeneralDataGenerator, GeneralPredictGenerator, SeqDataGenerator, seq_padding, \
  ELMoPredictGenerator
from loggers import Logger, BaseLogger, KFoldLogger
from utils import build_matrix, get_weights, get_weight_2, get_weights2, get_weight_ng, get_weights_new, \
  get_weights_new_array
from models.lstm_model import get_lstm_model, get_lstm_iden_model, get_lstm_attention_model, get_lstm_elmo_model, \
  get_lstm_adv_model
from tqdm import tqdm
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from evaluator import JigsawEvaluator
from bilm import Batcher as BT
from focal_loss import binary_crossentropy_with_ranking

model_name = 'bilstm_iden_5fold_ad_w6'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))
GEN = 1

IDENTITY_COLUMNS = [
  'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
  'muslim', 'black', 'white', 'psychiatric_or_mental_illness'
]
AUX_COLUMNS = ['target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']
AUX_AUG_COLUMNS = ['toxic', 'severe_toxic', 'obscene', 'identity_hate', 'insult', 'threat']
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


def load_model_weights(model, path):
  """ keras model
  :param model: keras model
  :param path weights存储路径
  :return:
  """
  with open(path, 'rb') as f:
    weights = pickle.load(f)
  for layer in model.layers:
    layer_name = layer.name
    if layer_name.startswith("embedding_"):
      continue
    if 'concatenate' in layer_name:
      key = ''
      for k in weights.keys():
        if 'concatenate' in k:
          key = k
    else:
      key = layer_name
    if not key in weights:
      continue
    layer.set_weights(weights[key])
  return model


class ExponentialMovingAverage:
  """对模型权重进行指数滑动平均。
  用法：在model.compile之后、第一次训练之前使用；
  先初始化对象，然后执行inject方法。
  """

  def __init__(self, model, momentum=0.9999):
    self.momentum = momentum
    self.model = model
    self.ema_weights = [K.zeros(K.shape(w)) for w in model.weights]

  def inject(self):
    """添加更新算子到model.metrics_updates。
    """
    self.initialize()
    for w1, w2 in zip(self.ema_weights, self.model.weights):
      op = K.moving_average_update(w1, w2, self.momentum)
      self.model.metrics_updates.append(op)

  def initialize(self):
    """ema_weights初始化跟原模型初始化一致。
    """
    self.old_weights = K.batch_get_value(self.model.weights)
    K.batch_set_value(zip(self.ema_weights, self.old_weights))

  def apply_ema_weights(self):
    """备份原模型权重，然后将平均权重应用到模型上去。
    """
    self.old_weights = K.batch_get_value(self.model.weights)
    ema_weights = K.batch_get_value(self.ema_weights)
    K.batch_set_value(zip(self.model.weights, ema_weights))

  def reset_old_weights(self):
    """恢复模型到旧权重。
    """
    K.batch_set_value(zip(self.model.weights, self.old_weights))


def val():
  df = pd.read_csv('new_processed_data/train_tok.csv')
  iden_df = pd.read_csv('processed_data/train_tok_iden.csv')

  import pickle
  embedding_matrix = pickle.load(open('new_processed_data/emb.pkl', 'rb'))
  texts = pickle.load(open('new_processed_data/texts.pkl', 'rb'))

  val_gen = GeneralPredictGenerator(text=texts, batch_size=512)

  model = get_lstm_model(embedding_matrix, len(AUX_COLUMNS))

  load_model_weights(model, 'save_models/weights.8-93.96.lstm_dp0.5_n_ema.pkl')

  result = model.predict_generator(val_gen.__iter__(), len(val_gen))[0]

  df['lstm_result'] = result

  train_ind, val_ind = train_test_split(range(len(texts)), random_state=59, test_size=0.055)

  val_df = df.iloc[val_ind]

  df.to_csv('new_processed_data/train_res.csv')
  val_df.to_csv('new_processed_data/val_res.csv')


def train_split_aug():
  df = pd.read_csv('new_processed_data/train_tok.csv')
  iden_df = pd.read_csv('processed_data/train_tok_iden.csv')
  iden_aug_df = pd.read_csv('new_processed_data/train_iden_last.csv')
  toxic_aug_df = pd.read_csv('new_processed_data/train_back_toxic.csv')

  ### text
  a_texts = df['comment_text'].values
  i_texts = iden_aug_df['comment_text'].values
  t_texts = toxic_aug_df['comment_text'].values
  texts = np.concatenate([a_texts, i_texts, t_texts])

  ### label
  a_label = df['target'].values
  i_label = iden_aug_df['toxic'].values
  t_label = toxic_aug_df['toxic'].values
  labels = np.concatenate([a_label, i_label, t_label])

  ### aux label
  a_aux = df[AUX_COLUMNS].values
  i_aux = iden_aug_df[AUX_AUG_COLUMNS].values
  t_aux = toxic_aug_df[AUX_AUG_COLUMNS].values
  aux = np.concatenate([a_aux, i_aux, t_aux])

  ### idts
  val_idts = df[IDENTITY_COLUMNS].fillna(0).values
  a_idts = iden_df[IDENTITY_COLUMNS].fillna(0).values
  i_idts = iden_aug_df[IDENTITY_COLUMNS].fillna(0).values
  t_idts = toxic_aug_df[IDENTITY_COLUMNS].fillna(0).values
  idts = np.concatenate([a_idts, i_idts, t_idts])

  del df
  del iden_df
  del iden_aug_df
  del toxic_aug_df

  tokenizer = text.Tokenizer(filters='', lower=False)
  tokenizer.fit_on_texts(list(texts))
  texts = tokenizer.texts_to_sequences(texts)
  texts = [t[:1024] for t in texts]

  crawl_matrix, unknown_words_crawl = build_matrix(tokenizer.word_index, 'embedding/crawl-300d-2M.pkl')
  print('n unknown words (crawl): ', len(unknown_words_crawl))

  glove_matrix, unknown_words_glove = build_matrix(tokenizer.word_index, 'embedding/glove.840B.300d.pkl')
  print('n unknown words (glove): ', len(unknown_words_glove))

  max_features = len(tokenizer.word_index) + 1
  print('Vocab Size:', max_features)

  embedding_matrix = np.concatenate([crawl_matrix, glove_matrix], axis=-1)
  print('Embedding shape:', embedding_matrix.shape)

  del crawl_matrix
  del glove_matrix
  gc.collect()

  import pickle
  pickle.dump(embedding_matrix, open('new_processed_data/aug_emb.pkl', 'wb'))
  pickle.dump(tokenizer.word_index, open('new_processed_data/aug_word_index.pkl', 'wb'))
  pickle.dump(texts, open('new_processed_data/aug_texts.pkl', 'wb'))

  train_ind, val_ind = train_test_split(range(len(a_texts)), random_state=59, test_size=0.055)

  train_texts = [texts[i] for i in train_ind] + texts[len(a_texts):]
  val_texts = [texts[i] for i in val_ind]

  train_labels, val_labels = np.concatenate([labels[train_ind], labels[len(a_texts):]]), labels[val_ind]
  train_aux_labels = np.concatenate([aux[train_ind], aux[len(a_texts):]])
  train_iden, val_iden = np.concatenate([idts[train_ind], idts[len(a_texts):]]), val_idts[val_ind]

  train_weight = get_weights_new_array(train_iden, train_labels)
  lw = 1 / np.mean(train_weight)

  train_gen = GeneralDataGenerator(inputs=[train_texts], outputs=[train_labels, train_aux_labels],
                                   sample_weights=[train_weight, np.ones_like(train_weight)],
                                   batch_size=512)
  val_gen = GeneralPredictGenerator(text=val_texts, batch_size=512)

  model = get_lstm_model(embedding_matrix, len(AUX_COLUMNS))

  opt = Adam(1e-3)

  model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[lw, 1.])
  model.summary()

  EMAer = ExponentialMovingAverage(model)
  EMAer.inject()

  logger = KFoldLogger('lstm_dp0.5_ema_aug', val_gen, val_true=val_labels, val_iden=val_iden, patience=10,
                       lr_patience=5)

  model.fit_generator(train_gen.__iter__(), len(train_gen), epochs=15, callbacks=[logger], verbose=1)


def train_split():
  df = pd.read_csv('new_processed_data/train_tok.csv')
  iden_df = pd.read_csv('processed_data/train_tok_iden.csv')

  labels_aux = df[AUX_COLUMNS].values
  identities = iden_df[IDENTITY_COLUMNS].fillna(0).values
  labels = df[TARGET_COLUMN].values
  weights = get_weights2(iden_df)

  # labels = (labels >= 0.5).astype(np.float)
  # labels_aux = (labels_aux >= 0.5).astype(np.float)

  import pickle
  embedding_matrix = pickle.load(open('new_processed_data/emb.pkl', 'rb'))
  # bpe_embedding_matrix = pickle.load(open('new_processed_data/bpe_embedding_matrix.pkl', 'rb'))
  texts = pickle.load(open('new_processed_data/texts.pkl', 'rb'))

  # bpe_embedding_matrix = np.concatenate([bpe_embedding_matrix, bpe_embedding_matrix], axis=1)
  # embedding_matrix += bpe_embedding_matrix
  del iden_df
  del df

  train_ind, val_ind = train_test_split(range(len(texts)), random_state=59, test_size=0.055)
  train_texts = [texts[i][:1024] for i in train_ind]
  val_texts = [texts[i][:1024] for i in val_ind]

  train_labels, val_labels = labels[train_ind], labels[val_ind]
  train_weight = weights[train_ind]
  # train_weight = train_weight / np.mean(train_weight)
  lw = 1 / np.mean(train_weight)
  train_aux_labels = labels_aux[train_ind]
  train_iden, val_iden = identities[train_ind], identities[val_ind]

  train_gen = GeneralDataGenerator(inputs=[train_texts], outputs=[train_labels, train_aux_labels],
                                   sample_weights=[train_weight, np.ones_like(train_weight)],
                                   batch_size=512)
  val_gen = GeneralPredictGenerator(text=val_texts, batch_size=512)

  # model = get_dcnn_model(embedding_matrix, len(AUX_COLUMNS))
  # model = get_lstm_model(embedding_matrix, len(AUX_COLUMNS))
  model = get_lstm_model(embedding_matrix, len(AUX_COLUMNS))
  # model.compile(loss=[binary_crossentropy_with_ranking, 'binary_crossentropy'], optimizer='adam')
  # opt = RMSprop(lr=1e-3)
  opt = Adam(1e-3)
  # lr = 1e-3
  # weight_decay = 0.01
  # bsz = 32
  # decay_steps = 10 * len(train_gen)
  # warmup_steps = int(0.1 * decay_steps)
  #
  # opt = AdamWarmup(
  #     decay_steps=decay_steps,
  #     warmup_steps=warmup_steps,
  #     lr=lr,
  #     weight_decay=weight_decay
  # )
  # load_model_weights(model, 'save_models/weights.0.9380885160416297.dcnn_dp0.5_n_deep.pkl')

  model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[lw, 1.])
  model.summary()

  EMAer = ExponentialMovingAverage(model)
  EMAer.inject()

  logger = KFoldLogger('lstm_w0_final', val_gen, val_true=val_labels, val_iden=val_iden, patience=10,
                       lr_patience=5)

  model.fit_generator(train_gen.__iter__(), len(train_gen), epochs=15, callbacks=[logger], verbose=1)


def train_elmo():
  # 489603
  # <S> 489604
  # </S> 489605
  df = pd.read_csv('new_processed_data/train_tok.csv')
  iden_df = pd.read_csv('processed_data/train_tok_iden.csv')

  datadir = os.path.join('elmo_model')
  options_file = os.path.join(datadir, 'elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json')
  weight_file = os.path.join(datadir, 'elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5')

  texts = df['comment_text'].values
  labels_aux = df[AUX_COLUMNS].values
  identities = df[IDENTITY_COLUMNS].fillna(0).values
  labels = df[TARGET_COLUMN].values
  weights = get_weights_new(iden_df)

  # import pickle
  # word_index = pickle.load(open('new_processed_data/word_index.pkl', 'rb'))
  # with open('new_processed_data/vocab.txt', 'w', encoding='utf8') as f:
  #     for k in word_index.keys():
  #         f.write(f'{k}\n')
  #     f.write('</S>')
  #     f.write('<S>')
  import pickle
  embedding_matrix = pickle.load(open('new_processed_data/emb.pkl', 'rb'))
  texts_ids = pickle.load(open('new_processed_data/texts.pkl', 'rb'))

  batcher = BT('new_processed_data/vocab.txt', 50)

  del iden_df
  del df

  train_ind, val_ind = train_test_split(range(len(texts)), random_state=59, test_size=0.055)
  # FOR ELMO
  train_texts, val_texts = texts[train_ind], texts[val_ind]
  train_texts = [s.split(' ')[:512] for s in train_texts]
  val_texts = [s.split(' ')[:512] for s in val_texts]

  # FOR W2v
  train_texts_ids = [texts_ids[i] for i in train_ind]
  val_texts_ids = [texts_ids[i] for i in val_ind]

  train_texts_ids = [ti[:512] for ti in train_texts_ids]
  val_texts_ids = [ti[:512] for ti in val_texts_ids]

  train_labels, val_labels = labels[train_ind], labels[val_ind]
  train_weight = weights[train_ind]
  lw = 1 / np.mean(train_weight)
  train_aux_labels = labels_aux[train_ind]
  train_iden, val_iden = identities[train_ind], identities[val_ind]

  pad_fn = [lambda x: seq_padding(x, truncate=False), batcher.batch_sentences]
  train_gen = GeneralDataGenerator(inputs=[train_texts_ids, train_texts], outputs=[train_labels, train_aux_labels],
                                   sample_weights=[train_weight, np.ones_like(train_weight)],
                                   batch_size=32, pad_fn=pad_fn)
  val_gen = ELMoPredictGenerator(text_ids=val_texts_ids, text=val_texts, pad_fn=pad_fn, batch_size=32)

  model = get_lstm_elmo_model(embedding_matrix, weight_file, options_file, 1024, len(AUX_COLUMNS))

  # lr = 1e-3
  # weight_decay = 0.01
  # bsz = 32
  # decay_steps = 3 * len(train_gen)
  # warmup_steps = int(0.05 * decay_steps)
  #
  # optimizer = AdamWarmup(
  #     decay_steps=decay_steps,
  #     warmup_steps=warmup_steps,
  #     lr=lr,
  #     weight_decay=weight_decay
  # )
  optimizer = Adam(1e-3)
  load_model_weights(model, 'save_models/weights.3-93.597.elmo_w2v_lstm2_dp05.pkl')
  model.compile(loss='binary_crossentropy', optimizer=optimizer, loss_weights=[lw, 1.])
  model.summary()

  logger = KFoldLogger('elmo_w2v_lstm2_dp05', val_gen, val_true=val_labels, val_iden=val_iden)

  model.fit_generator(train_gen.__iter__(), len(train_gen), epochs=5, callbacks=[logger], initial_epoch=3)


def train_fold():
  df = pd.read_csv('new_processed_data/train_tok.csv')
  iden_df = pd.read_csv('processed_data/train_tok_iden.csv')

  # tqdm.pandas()
  # texts = df['comment_text'].fillna('none').progress_apply(lambda s: ' '.join(tk.tokenize(s)))
  # df['comment_text'] = texts
  # df.to_csv('new_processed_data/train_tok.csv', index=False)

  # texts = df['comment_text'].values
  labels_aux = df[AUX_COLUMNS].values
  identities = df[IDENTITY_COLUMNS].values
  labels = df[TARGET_COLUMN].values

  weights = get_weights2(iden_df)

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
  #
  # del crawl_matrix
  # del glove_matrix
  # gc.collect()

  import pickle
  # pickle.dump(embedding_matrix, open('new_processed_data/emb.pkl', 'wb'))
  # pickle.dump(tokenizer.word_index, open('new_processed_data/word_index.pkl', 'wb'))
  # pickle.dump(texts, open('new_processed_data/texts.pkl', 'wb'))

  embedding_matrix = pickle.load(open('new_processed_data/emb.pkl', 'rb'))
  texts = pickle.load(open('new_processed_data/texts.pkl', 'rb'))

  iden_df = pd.read_csv('processed_data/train_tok_iden.csv')
  pre_identities = iden_df[['pre_' + c for c in IDENTITY_COLUMNS]].values
  del iden_df

  kfold = StratifiedKFold(n_splits=5, random_state=59)

  batch_size = 512
  cnt = 0
  oof = np.zeros(len(texts))

  # MOD
  # for i in range(len(labels)):
  #     is_iden = np.sum((identities[i] >= 0.5).astype(int)) >= 1
  #     l = 0 if labels[i] < 0.5 else 1
  #     labels[i] = l if is_iden else labels[i]
  # labels = (labels >= 0.5).astype(int)
  # labels_aux = np.asarray([[int(y >= 0.5) for y in x] for x in labels_aux])

  for train_ind, test_ind in kfold.split(np.zeros(len(labels)), labels >= 0.5):
    print(f"Fold {cnt}")
    train_texts, test_texts = [], []
    for i in train_ind:
      train_texts.append(texts[i])
    for i in test_ind:
      test_texts.append(texts[i])
    train_label, test_label = labels[train_ind], labels[test_ind]
    train_aux_label, test_aux_label = labels_aux[train_ind], labels_aux[test_ind]
    train_pre_iden, _ = pre_identities[train_ind], pre_identities[test_ind]
    train_iden, test_iden = identities[train_ind], pre_identities[test_ind]
    train_weight, _ = weights[train_ind], weights[test_ind]

    train_weight = train_weight / np.mean(train_weight)

    model, save_model = get_lstm_model(embedding_matrix, len(AUX_COLUMNS))
    train_gen = GeneralDataGenerator(inputs=[train_texts], outputs=[train_label, train_aux_label],
                                     sample_weights=[train_weight, np.ones_like(train_weight)],
                                     batch_size=batch_size)
    val_gen = GeneralPredictGenerator(text=test_texts)

    logger = KFoldLogger(model_name + f'_{cnt}', val_gen, val_true=test_label, val_iden=test_iden)

    model.fit_generator(train_gen.__iter__(), len(train_gen), epochs=10, callbacks=[logger], verbose=2)

    oof[test_ind] = logger.pred
    cnt += 1

    train_gen.close()
    del train_gen
    del val_gen

  evaluator = JigsawEvaluator(labels, identities)
  final_auc, overall_auc, sub_auc, bpsn_auc, bnsp_auc, bias_metrics = evaluator.get_final_metric(oof)
  print('Final AUC:{}\nOverall AUC:{}\nSub AUC:{}\nBPSN AUC:{}\nBNSP AUC:{}\n'.format(final_auc, overall_auc,
                                                                                      sub_auc, bpsn_auc,
                                                                                      bnsp_auc))
  print('Detail Bias:\n', bias_metrics)
  with open('Result.csv', 'a', encoding='utf8') as f:
    f.write('{},{},{},{},{},{}\n'.format(model_name + '_oof', final_auc, overall_auc, sub_auc, bpsn_auc,
                                         bnsp_auc))


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
  # weights = get_weight_2(df)
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

  # train_indxs, val_indxs = train_test_split(range(len(df)), random_state=59, test_size=0.055)
  # train_weight = weights[train_indxs]
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
  # pickle.dump(train_weight, open('processed_data/train_weight_2.pkl', 'wb'))
  # val_df.to_csv('processed_data/val.csv', index=False)

  embedding_matrix = pickle.load(open('processed_data3/emb.pkl', 'rb'))
  train_texts = pickle.load(open('processed_data3/train_texts.pkl', 'rb'))
  val_texts = pickle.load(open('processed_data3/val_texts.pkl', 'rb'))
  train_label = pickle.load(open('processed_data3/train_label.pkl', 'rb'))
  train_aux_label = pickle.load(open('processed_data3/train_aux_label.pkl', 'rb'))
  train_weight = pickle.load(open('processed_data/train_weight_iden.pkl', 'rb'))
  val_df = pd.read_csv('processed_data3/val.csv')
  print(train_weight.shape)
  print(train_label.shape)

  # train_label = [int(x > 0.5) for x in train_label]
  # train_aux_label = [[int(y > 0.5) for y in x] for x in train_aux_label]
  if GEN != 0:
    train_gen = GeneralDataGenerator(inputs=[train_texts], outputs=[train_label, train_aux_label],
                                     sample_weights=[train_weight, np.ones_like(train_weight)],
                                     batch_size=batch_size)
    # train_gen = GeneralDataGenerator(inputs=[train_texts], outputs=[train_label, train_aux_label],
    #                                  sample_weights=None,
    #                                  batch_size=batch_size)
  else:
    train_gen = SeqDataGenerator(train_texts, train_label, train_aux_label, train_weight)
  val_gen = GeneralPredictGenerator(text=val_texts)

  # model, save_model = get_lstm_model(embedding_matrix, len(AUX_COLUMNS))
  model, save_model = get_lstm_model(embedding_matrix, len(AUX_COLUMNS))

  logger = Logger(save_model, model_name, val_gen, val_df)
  # model.load_weights('save_models/weights.bilstm_iden.h5')
  model.fit_generator(train_gen.__iter__(), len(train_gen), epochs=5, callbacks=[logger])


def train_indetity():
  df = pd.read_csv('new_processed_data/train_tok.csv')
  iden_indexs = df[df['identity_annotator_count'] > 0].index.values
  iden_labels = df.iloc[iden_indexs][IDENTITY_COLUMNS].values
  iden_texts = df.iloc[iden_indexs]['comment_text'].values
  # texts = df['comment_text'].values
  # tokenizer = text.Tokenizer(filters='', lower=False)
  # tokenizer.fit_on_texts(list(texts))
  import pickle
  # pickle.dump(tokenizer.word_index, open('processed_data/word_index.pkl'))

  embedding_matrix = pickle.load(open('new_processed_data/emb.pkl', 'rb'))
  word_index = pickle.load(open('new_processed_data/word_index.pkl', 'rb'))

  def text_to_ids(t):
    t = t.split(' ')
    t = [word_index[w] for w in t]
    return t

  iden_texts = [text_to_ids(t) for t in iden_texts]
  iden_labels = [[int(la >= 0.5) for la in label] for label in iden_labels]

  train_text, val_text, train_label, val_label = train_test_split(iden_texts, iden_labels, test_size=0.055,
                                                                  random_state=59)

  train_gen = GeneralDataGenerator(inputs=[train_text], outputs=[train_label],
                                   sample_weights=None,
                                   batch_size=512)
  val_gen = GeneralPredictGenerator(text=val_text)

  logger = BaseLogger(val_gen, val_label)

  model = get_lstm_iden_model(embedding_matrix, len(IDENTITY_COLUMNS))

  model.fit_generator(train_gen.__iter__(), len(train_gen), epochs=10
                      , callbacks=[logger])


if __name__ == '__main__':
  # train()
  # train_indetity()
  # train_fold()
  # train_elmo()
  train_split()
  # val()
  # train_split_aug()
