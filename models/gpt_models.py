from keras.layers import Dense, Lambda
from keras.models import Model
import keras.backend as K
from keras_bert.layers import Extract
from keras_gpt_2 import load_trained_model_from_checkpoint, get_bpe_from_files, generate


def seq_gather(x):
    """seq是[None, seq_len, s_size]的格式，
    idxs是[None, 1]的格式，在seq的第i个序列中选出第idxs[i]个向量，
    最终输出[None, s_size]的向量。
    """
    seq, idxs = x
    idxs = K.cast(idxs, 'int32')
    batch_idxs = K.arange(0, K.shape(seq)[0])
    batch_idxs = K.expand_dims(batch_idxs, 1)
    idxs = K.concatenate([batch_idxs, idxs], 1)
    return K.tf.gather_nd(seq, idxs)


def get_gpt_model(config_path, checkpoint_path):
    model = load_trained_model_from_checkpoint(config_path, checkpoint_path)
    inputs = model.inputs[0]
    mask = Lambda(lambda x: K.reshape(K.sum(K.cast(K.greater(x, 0), 'float32'), axis=-1), [K.shape(x)[0], 1]) - 1,
                  name='Mask')(inputs)
    # mask = Lambda(lambda x: print(K.shape(x)),
    #               name='Mask')(inputs)
    layer = model.get_layer(name='Norm').output
    layer = Lambda(seq_gather, name='Gather')([layer, mask])
    predict = Dense(1, activation='sigmoid', name='Predict-Dense')(layer)
    aux = Dense(6, activation='sigmoid', name='Predict-Aux')(layer)

    model = Model(inputs=inputs, outputs=[predict, aux])
    model.summary()
    return model
