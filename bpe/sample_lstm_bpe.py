!pip
install
bpemb

import numpy as np
import pandas as pd
from bpemb import BPEmb
from keras.callbacks import LearningRateScheduler
from keras.layers import CuDNNLSTM, LSTM, Bidirectional, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, add, concatenate
from keras.models import Model
from keras_preprocessing.sequence import pad_sequences
from pathlib import Path

NUM_MODELS = 2
BATCH_SIZE = 512
LSTM_UNITS = 128
DENSE_HIDDEN_UNITS = 4 * LSTM_UNITS
EPOCHS = 4
IDENTITY_COLUMNS = [
    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
    'muslim', 'black', 'white', 'psychiatric_or_mental_illness'
]
AUX_COLUMNS = ['target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']
TEXT_COLUMN = 'comment_text'
TARGET_COLUMN = 'target'


def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')


def load_embeddings(path):
    with open(path) as f:
        return dict(get_coefs(*line.strip().split(' ')) for line in f)


def build_matrix(word_index, path):
    embedding_index = load_embeddings(path)
    embedding_matrix = np.zeros((len(word_index) + 1, 300))
    for word, i in word_index.items():
        try:
            embedding_matrix[i] = embedding_index[word]
        except KeyError:
            pass
    return embedding_matrix


def build_model(embedding_matrix, num_aux_targets):
    words = Input(shape=(None,))
    x = Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False)(words)
    x = SpatialDropout1D(0.2)(x)
    x = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True))(x)
    x = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True))(x)

    hidden = concatenate([
        GlobalMaxPooling1D()(x),
        GlobalAveragePooling1D()(x),
    ])
    hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])
    hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])
    result = Dense(1, activation='sigmoid')(hidden)
    aux_result = Dense(num_aux_targets, activation='sigmoid')(hidden)

    model = Model(inputs=words, outputs=[result, aux_result])
    model.compile(loss='binary_crossentropy', optimizer='adam')

    return model


nrows = None
train_df = pd.read_csv("../input/mydata/train_preprocessed.csv", nrows=nrows)
test_df = pd.read_csv("../input/mydata/test_preprocessed.csv", nrows=nrows)
# train_df = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv')
# test_df = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv')


x_train = train_df[TEXT_COLUMN].astype(str)
y_train = train_df[TARGET_COLUMN].values
y_aux_train = train_df[AUX_COLUMNS].values
x_test = test_df[TEXT_COLUMN].astype(str)

for column in IDENTITY_COLUMNS + [TARGET_COLUMN]:
    train_df[column] = np.where(train_df[column] >= 0.5, True, False)

vocab_size = 100 * 1000
embedding_dim = 100
max_sequence_len = 220

bpemb_en_100k = BPEmb(lang="en", vs=vocab_size, dim=embedding_dim,
                      cache_dir=Path("."))  # 40 M；词表越大切分越少


def get_x_data(sentences):
    sentences_ids = bpemb_en_100k.encode_ids(sentences)  # 使用bpe
    x = pad_sequences(sentences_ids, maxlen=max_sequence_len)
    return x


x_train = get_x_data(x_train)
x_test = get_x_data(x_test)

sample_weights = np.ones(len(x_train), dtype=np.float32)
sample_weights += train_df[IDENTITY_COLUMNS].sum(axis=1)
sample_weights += train_df[TARGET_COLUMN] * (~train_df[IDENTITY_COLUMNS]).sum(axis=1)
sample_weights += (~train_df[TARGET_COLUMN]) * train_df[IDENTITY_COLUMNS].sum(axis=1) * 5
sample_weights /= sample_weights.mean()

checkpoint_predictions = []
weights = []

for model_idx in range(NUM_MODELS):
    model = build_model(bpemb_en_100k.vectors, y_aux_train.shape[-1])
    for global_epoch in range(EPOCHS):
        model.fit(
            x_train,
            [y_train, y_aux_train],
            batch_size=BATCH_SIZE,
            epochs=1,
            verbose=2,
            sample_weight=[sample_weights.values, np.ones_like(sample_weights)],
            callbacks=[
                LearningRateScheduler(lambda _: 1e-3 * (0.55 ** global_epoch))
            ]
        )
        checkpoint_predictions.append(model.predict(x_test, batch_size=2048)[0].flatten())
        weights.append(2 ** global_epoch)
        try:
            model.save("{}.hdf5".format(global_epoch))
        except:
            pass

predictions = np.average(checkpoint_predictions, weights=weights, axis=0)

submission = pd.DataFrame.from_dict({
    'id': test_df.id,
    'prediction': predictions
})
submission.to_csv('submission.csv', index=False)
