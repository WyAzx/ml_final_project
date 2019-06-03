import gc
import pandas as pd
from keras.callbacks import LearningRateScheduler
from keras.layers import *
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

from Capsule_Keras import Capsule

maxlen = 220
max_features = 100000
embed_size = 300
batch_size = 64
NUM_MODELS = 2
EPOCHS = 5

IDENTITY_COLUMNS = [
    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
    'muslim', 'black', 'white', 'psychiatric_or_mental_illness'
]

AUX_COLUMNS = ['target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']
TEXT_COLUMN = 'comment_text'
TARGET_COLUMN = 'target'
TEXT_COL = "comment_text"

train_df = pd.read_csv("../data/train_preprocessed.csv")
test_df = pd.read_csv("../data/test_preprocessed.csv")
# train_df = pd.read_csv("../input/train.csv")
# test_df = pd.read_csv("../input/test.csv")

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
y_aux_train = train_df[AUX_COLUMNS].values

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
    # capsule = Dropout(0.25)(capsule)
    #     capsule = LeakyReLU()(capsule)

    #     x = Flatten()(x)
    # output1 = Dense(1, activation='sigmoid', name="out1")(Dropout(0.25)(capsule))
    # output2 = Dense(1, activation='sigmoid', name="out2")(Dropout(0.3)(capsule))
    # output3 = Dense(1, activation='sigmoid', name="out3")(Dropout(0.5)(capsule))
    # model = Model(inputs=input1, outputs=[output1, output2, output3])
    y_output = Dense(1, activation='sigmoid', name="out")(Dropout(0.25)(capsule))
    y_auc_output = Dense(len(AUX_COLUMNS), activation='sigmoid')(capsule)
    model = Model(inputs=input1, outputs=[y_output, y_auc_output])  # muti task
    return model


for column in IDENTITY_COLUMNS + [TARGET_COLUMN]:
    train_df[column] = np.where(train_df[column] >= 0.5, True, False)

sample_weights = np.ones(len(X_train), dtype=np.float32)
sample_weights += train_df[IDENTITY_COLUMNS].sum(axis=1)
sample_weights += train_df[TARGET_COLUMN] * (~train_df[IDENTITY_COLUMNS]).sum(axis=1)
sample_weights += (~train_df[TARGET_COLUMN]) * train_df[IDENTITY_COLUMNS].sum(axis=1) * 5
sample_weights /= sample_weights.mean()

checkpoint_predictions = []
weights = []

for model_idx in range(NUM_MODELS):
    model = get_model()
    model.compile(loss='binary_crossentropy', optimizer='adam')
    for global_epoch in range(EPOCHS):
        model.fit(
            X_train,
            [Y_train, y_aux_train],
            batch_size=128,
            epochs=1,
            verbose=2,
            sample_weight=[sample_weights.values, np.ones_like(sample_weights)],
            callbacks=[
                LearningRateScheduler(lambda _: 1e-3 * (0.55 ** global_epoch))
            ]
        )
        checkpoint_predictions.append(model.predict(X_test, batch_size=2048)[0].flatten())
        weights.append(2 ** global_epoch)

predictions = np.average(checkpoint_predictions, weights=weights, axis=0)

submission = pd.DataFrame.from_dict({
    'id': test_df.id,
    'prediction': predictions
})
submission.to_csv('submission.csv', index=False)
