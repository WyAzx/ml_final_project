import numpy as np
from bpemb import BPEmb
from keras import Input, Model
from keras.layers import Embedding, Convolution1D, MaxPool1D, concatenate, Flatten, Dropout, Dense
import pandas as pd
from keras_preprocessing.sequence import pad_sequences


def get_text_cnn(vocab_size, max_sequence_len, embedding_dim, num_classes, embedding_matrix=None):
    # 模型结构：词嵌入-卷积池化*3-拼接-全连接-dropout-全连接
    main_input = Input(shape=(max_sequence_len,))
    # 词嵌入（使用预训练的词向量）
    embedder = Embedding(vocab_size, embedding_dim, input_length=max_sequence_len,
                         weights=np.asarray([embedding_matrix]),
                         trainable=False)
    embed = embedder(main_input)
    # 词窗大小分别为3,4,5
    cnn1 = Convolution1D(256, 3, padding='same', strides=1, activation='relu')(embed)
    cnn1 = MaxPool1D(pool_size=4)(cnn1)
    cnn2 = Convolution1D(256, 4, padding='same', strides=1, activation='relu')(embed)
    cnn2 = MaxPool1D(pool_size=4)(cnn2)
    cnn3 = Convolution1D(256, 5, padding='same', strides=1, activation='relu')(embed)
    cnn3 = MaxPool1D(pool_size=4)(cnn3)
    # 合并三个模型的输出向量
    cnn = concatenate([cnn1, cnn2, cnn3], axis=-1)
    flat = Flatten()(cnn)
    drop = Dropout(0.2)(flat)
    output = Dense(num_classes, activation='sigmoid')(drop)
    model = Model(inputs=main_input, outputs=output)
    return model


class DataLoader():
    def __init__(self, vocab_size, embedding_dim, max_sequence_len):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_sequence_len = max_sequence_len
        self.bpemb_en_100k = BPEmb(lang="en", vs=self.vocab_size, dim=self.embedding_dim)  # 40 M；词表越大切分越少

    def get_x_data(self, sentences):
        sentences_ids = self.bpemb_en_100k.encode_ids(sentences)  # 使用bpe
        x = pad_sequences(sentences_ids, maxlen=self.max_sequence_len)
        return x

    def get_train_data(self):
        nrows = 100
        train_df = pd.read_csv("../data/train_preprocessed.csv", nrows=nrows)
        X_train = self.get_x_data(train_df["comment_text"])
        Y_train = train_df['target'].values
        x, y = np.asarray(X_train), np.asarray(Y_train)
        print(x.shape, y.shape)
        return x, y

    def get_test_data(self):
        nrows = 100
        test_df = pd.read_csv("../data/test_preprocessed.csv", nrows=nrows)
        X_test = self.get_x_data(test_df["comment_text"])
        return np.asarray(X_test), test_df


def train():
    vocab_size = 100 * 1000
    embedding_dim = 100
    max_sequence_len = 512
    num_classes = 1
    data_loader = DataLoader(vocab_size, embedding_dim, max_sequence_len)
    model = get_text_cnn(vocab_size, max_sequence_len, embedding_dim,
                         num_classes,
                         embedding_matrix=data_loader.bpemb_en_100k.vectors)  # 使用bpe

    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy'],
        # metrics=[keras_metrics.precision(), keras_metrics.recall()],
    )
    X_train, Y_train = data_loader.get_train_data()
    model.fit(X_train, Y_train,
              batch_size=32, validation_split=0.1,
              verbose=1, shuffle=True
              )
    model.save("text_cnn.hdf5")
    X_test, test_df = data_loader.get_test_data()
    preds = model.predict(X_test)
    submission = pd.DataFrame.from_dict({
        'id': test_df.id,
        'prediction': preds.flatten().tolist()
    })
    submission.to_csv('submission.csv', index=False)


if __name__ == "__main__":
    train()
