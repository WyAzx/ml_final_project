from keras_bert.backend import keras
from keras_bert.backend import backend as K

from keras_bert.layers.extract import Extract
from keras_bert.loader import load_trained_model_from_checkpoint
from utils import BertConfig


def get_bert_base_model(bert_config: BertConfig):
    bert_model = load_trained_model_from_checkpoint(bert_config.config, bert_config.check_point, trainable=True,
                                                    seq_len=512)
    inputs = bert_model.inputs
    layer = bert_model.outputs[0]
    layer = Extract(index=0, name='Extract')(layer)
    predicate = keras.layers.Dense(1, activation='sigmoid', name='Predicate-Dense')(layer)

    model = keras.models.Model(inputs=inputs, outputs=[predicate])
    model.summary()
    return model


def get_bert_multi_model(bert_config: BertConfig):
    bert_model = load_trained_model_from_checkpoint(bert_config.config, bert_config.check_point, trainable=True,
                                                    seq_len=512)
    inputs = bert_model.inputs
    layer = bert_model.outputs[0]
    layer = Extract(index=0, name='Extract')(layer)
    predict = keras.layers.Dense(1, activation='sigmoid', name='Predict-Dense')(layer)
    aux = keras.layers.Dense(6, activation='sigmoid', name='Predict-Aux')(layer)

    model = keras.models.Model(inputs=inputs, outputs=[predict, aux])
    model.summary()
    return model


def get_bert_multi_layers_model(bert_config: BertConfig):
    bert_model = load_trained_model_from_checkpoint(bert_config.config, bert_config.check_point, trainable=True,
                                                    seq_len=512, output_layer_num=4)
    inputs = bert_model.inputs
    layer = bert_model.outputs[0]
    layer = Extract(index=0, name='Extract')(layer)
    layer = keras.layers.Dense(512, activation='relu', name='Dense')(layer)
    predict = keras.layers.Dense(1, activation='sigmoid', name='Predict-Dense')(layer)
    aux = keras.layers.Dense(6, activation='sigmoid', name='Predict-Aux')(layer)

    model = keras.models.Model(inputs=inputs, outputs=[predict, aux])
    model.summary()
    return model
