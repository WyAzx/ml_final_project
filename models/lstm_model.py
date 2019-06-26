import keras.backend as K
import sys

from keras_grl import GradientReversal

sys.path.append('keras_layers')
from keras.layers import Embedding, Input, SpatialDropout1D, Bidirectional, CuDNNLSTM, concatenate, add, Dense, Lambda, \
    Masking, GlobalMaxPooling1D, Dropout
from keras.models import Model
from keras.models import Sequential
from focal_loss import focal_loss
from keras_han.layers import AttentionLayer
from keras.optimizers import Adam
from keras_elmo.elmo import ELMoEmbedding
import tensorflow as tf
from keras.optimizers import Adam

def seq_maxpool(x):
    seq, mask = x
    seq -= (1 - mask) * 1e10
    return K.max(seq, 1)


def seq_avgpool(x):
    seq, mask = x
    seq = seq * mask
    return K.mean(seq, 1)


def seq_mask(x):
    seq, mask = x
    seq = seq * mask
    return seq


def get_lstm_model(embedding_matrix, num_aux_targets):
    words = Input(shape=(None,), name='Text-Input')
    mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'), name='Mask')(words)
    emb = Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False)(words)
    # x = SpatialDropout1D(0.2, name='Dropout')(emb)
    x = Dropout(0.5)(emb)
    x = Bidirectional(CuDNNLSTM(128, return_sequences=True), name='BiLSTM-L1')(x)
    x = Bidirectional(CuDNNLSTM(128, return_sequences=True), name='BiLSTM-L2')(x)

    hidden = concatenate([
        Lambda(seq_maxpool, name='Max_Pool')([x, mask]),
        Lambda(seq_avgpool, name='Avg_Pool')([x, mask]),
    ])
    hidden = add([hidden, Dense(512, activation='relu', name='Dense-L1')(hidden)])
    hidden = add([hidden, Dense(512, activation='relu', name='Dense-L2')(hidden)])
    result = Dense(1, activation='sigmoid', name='Label-Predict')(hidden)
    aux_result = Dense(num_aux_targets, activation='sigmoid', name='Aux-Label-Predict')(hidden)

    # save_model = Model(inputs=emb, outputs=[result, aux_result])
    model = Model(inputs=[words], outputs=[result, aux_result])

    return model


def get_lstm_adv_model(embedding_matrix, num_aux_targets):
  words = Input(shape=(None,), name='Text-Input')
  mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'), name='Mask')(words)
  emb = Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False)(words)
  # x = SpatialDropout1D(0.2, name='Dropout')(emb)
  x = Dropout(0.5)(emb)
  x = Bidirectional(CuDNNLSTM(128, return_sequences=True), name='BiLSTM-L1')(x)
  x = Bidirectional(CuDNNLSTM(128, return_sequences=True), name='BiLSTM-L2')(x)

  hidden = concatenate([
    Lambda(seq_maxpool, name='Max_Pool')([x, mask]),
    Lambda(seq_avgpool, name='Avg_Pool')([x, mask]),
  ])
  hidden = add([hidden, Dense(512, activation='relu', name='Dense-L1')(hidden)])
  hidden = add([hidden, Dense(512, activation='relu', name='Dense-L2')(hidden)])
  result = Dense(1, activation='sigmoid', name='Label-Predict')(hidden)
  aux_result = Dense(num_aux_targets, activation='sigmoid', name='Aux-Label-Predict')(hidden)

  grl = GradientReversal(1, name='GRL')(hidden)
  adv_result = Dense(9, activation='sigmoid', name='Adv_Predict')(grl)

  # save_model = Model(inputs=emb, outputs=[result, aux_result])
  model = Model(inputs=[words], outputs=[result, aux_result, adv_result])

  return model


def get_lstm_attention_model(embedding_matrix, num_aux_targets):
    words = Input(shape=(None,), name='Text-Input')
    mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'), name='Mask')(words)
    emb = Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False, name='Train-Embedding')(words)
    x = SpatialDropout1D(0.2, name='Dropout')(emb)
    x = Bidirectional(CuDNNLSTM(128, return_sequences=True), name='BiLSTM-L1')(x)
    x = Bidirectional(CuDNNLSTM(128, return_sequences=True), name='BiLSTM-L2')(x)

    x = Lambda(seq_mask)([x, mask])
    x = Masking()(x)
    x = AttentionLayer(128, name='Attention')(x)
    result = Dense(1, activation='sigmoid', name='Label-Predict')(x)
    aux_result = Dense(num_aux_targets, activation='sigmoid', name='Aux-Label-Predict')(x)

    # save_model = Model(inputs=emb, outputs=[result, aux_result])
    model = Model(inputs=[words], outputs=[result, aux_result])
    optimizer = Adam(lr=2e-4)
    model.compile(loss='binary_crossentropy', optimizer=optimizer)

    return model, model


def get_lstm_elmo_model(embedding_matrix, weight_file, option_file, embedding_dim, num_aux_targets):
    words = Input(shape=(None,), name='Text-Input')

    mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'), name='Mask')(words)
    emb = Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False)(words)

    charaters = Input(shape=(None, 50), name='Char-Input', dtype=tf.int32)
    elmo_emb = ELMoEmbedding(option_file, weight_file, embedding_dim, do_ln=True, name='ELMo-Embedding')(charaters)

    concate_emb = concatenate([elmo_emb, emb])

    x = Dropout(0.5, name='Dropout')(concate_emb)
    x = Bidirectional(CuDNNLSTM(256, return_sequences=True), name='BiLSTM-L1')(x)
    x = Bidirectional(CuDNNLSTM(256, return_sequences=True), name='BiLSTM-L2')(x)
    x = Lambda(seq_maxpool, name='Max_Pool')([x, mask])
    result = Dense(1, activation='sigmoid', name='Label-Predict')(x)
    aux_result = Dense(num_aux_targets, activation='sigmoid', name='Aux-Label-Predict')(x)

    model = Model(inputs=[words, charaters], outputs=[result, aux_result])

    return model


def get_lstm_iden_model(embedding_matrix, num_iden):
    words = Input(shape=(None,), name='Text-Input')
    mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'), name='Mask')(words)
    emb = Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False)(words)
    x = Bidirectional(CuDNNLSTM(128, return_sequences=True), name='BiLSTM-L1')(emb)
    x = Bidirectional(CuDNNLSTM(128, return_sequences=True), name='BiLSTM-L2')(x)

    x = Lambda(seq_maxpool)([x, mask])
    # x = Masking()(x)
    # x = AttentionLayer(128, name='Attention')(x)
    predict = Dense(num_iden, activation='sigmoid')(x)

    model = Model(inputs=[words], outputs=[predict])
    model.compile(loss='binary_crossentropy', optimizer='adam')

    return model
