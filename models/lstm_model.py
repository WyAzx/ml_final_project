import keras.backend as K
from keras.layers import Embedding, Input, SpatialDropout1D, Bidirectional, CuDNNLSTM, concatenate, add, Dense, Lambda
from keras.models import Model
from keras.models import Sequential


def seq_maxpool(x):
    seq, mask = x
    seq -= (1 - mask) * 1e10
    return K.max(seq, 1)


def seq_avgpool(x):
    seq, mask = x
    seq = seq * mask
    return K.mean(seq, 1)


def get_lstm_model(embedding_matrix, num_aux_targets):
    words = Input(shape=(None,), name='Text-Input')
    mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'), name='Mask')(words)
    emb = Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False, name='Train-Embedding')(words)
    x = SpatialDropout1D(0.2, name='Dropout')(emb)
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
    model.compile(loss='binary_crossentropy', optimizer='adam')

    return model, model
