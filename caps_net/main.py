# import os
#
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 指定GPU

import gc
import pandas as pd
from Capsule_Keras import Capsule
from keras.layers import *
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

maxlen = 220
max_features = 100000
embed_size = 300
batch_size = 64

# cur_dir = os.path.dirname(os.path.abspath("__file__"))
# root_dir = os.path.dirname(cur_dir)
# # data_dir = os.path.join(root_dir, "data")
# data_dir = os.path.join(root_dir, "data")
# train_file = os.path.join(data_dir, "train_preprocessed.csv")
# test_file = os.path.join(data_dir, "test_preprocessed.csv")

train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")

TEXT_COL = "comment_text"

train_df[TEXT_COL] = train_df[TEXT_COL].fillna('')
test_df[TEXT_COL] = test_df[TEXT_COL].fillna('')
train_df = train_df[train_df['comment_text'].str.len() >= 1]

tokenizer = Tokenizer(num_words=max_features, lower=True)  # filters = ''
# tokenizer = text.Tokenizer(num_words=max_features)
print('fitting tokenizer')
tokenizer.fit_on_texts(list(train_df[TEXT_COL]) + list(test_df[TEXT_COL]))
word_index = tokenizer.word_index

X_train = tokenizer.texts_to_sequences(list(train_df[TEXT_COL]))
X_train = pad_sequences(X_train, maxlen=maxlen)

Y_train = train_df['target'].values

X_test = tokenizer.texts_to_sequences(list(test_df[TEXT_COL]))
X_test = pad_sequences(X_test, maxlen=maxlen)

del tokenizer
gc.collect()

print("data prepare Done ...")


def get_model():
    input1 = Input(shape=(maxlen,))
    embed_layer = Embedding(max_features,
                            embed_size,
                            input_length=maxlen)(input1)
    embed_layer = SpatialDropout1D(0.28)(embed_layer)

    x = Bidirectional(GRU(256,
                          activation='relu',
                          dropout=0.25,
                          recurrent_dropout=0.25,
                          return_sequences=True))(embed_layer)
    capsule = Capsule(
        num_capsule=10,
        dim_capsule=16,
        routings=3,
        share_weights=True)(x)

    capsule = Flatten()(capsule)
    capsule = Dropout(0.25)(capsule)
    #     capsule = LeakyReLU()(capsule)

    #     x = Flatten()(x)
    output = Dense(1, activation='sigmoid')(capsule)
    model = Model(inputs=input1, outputs=output)
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])
    model.summary()

    return model


model = get_model()
model.fit(X_train, Y_train,
          batch_size=batch_size,
          epochs=10,
          verbose=1,
          # validation_data=(x_test, y_test)
          validation_split=0.1)
Y_pred = model.predict(X_test)  # 用模型进行预测
test_df["prediction"] = Y_pred.flatten().tolist()
test_df[["id", "prediction"]].to_csv("./sample_submission.csv", index=False)
print("Done ...")
