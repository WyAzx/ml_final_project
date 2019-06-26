import keras
from keras import backend as K
from keras.engine.topology import Layer
import tensorflow as tf
import numpy as np
import sys
sys.path.append('bilm-tf')
from bilm import BidirectionalLanguageModel
from keras.layers import BatchNormalization

class ELMoEmbedding(Layer):

    def __init__(self, options_file, weight_file, embedding_dim, do_ln=False, **kwargs):
        self.elmo_model = None
        self.options_file = options_file
        self.embedding_dim = embedding_dim
        self.weight_file = weight_file
        self.trainable = True
        self.do_ln = do_ln
        super(ELMoEmbedding, self).__init__(**kwargs)

    def build(self, input_shape):
        self.elmo_model = BidirectionalLanguageModel(self.options_file, self.weight_file, max_batch_size=32)
        self.W = self.add_weight(
            name='W',
            shape=(3,),
            initializer=keras.initializers.get('zeros'),
            trainable=True
        )
        self.gamma = self.add_weight(
            name='gamma', shape=(1,),
            initializer=keras.initializers.get('ones'),
            trainable=True
        )
        super(ELMoEmbedding, self).build(input_shape)

    def call(self, x, mask=None):
        embeddings = self.elmo_model(x)

        # Get ops for computing LM embeddings and mask
        lm_embeddings = embeddings['lm_embeddings']
        mask = embeddings['mask']

        n_lm_layers = int(lm_embeddings.get_shape()[1])
        lm_dim = int(lm_embeddings.get_shape()[3])
        print(lm_dim)

        with tf.control_dependencies([lm_embeddings, mask]):
            # Cast the mask and broadcast for layer use.
            mask_float = tf.cast(mask, 'float32')
            broadcast_mask = tf.expand_dims(mask_float, axis=-1)

            def _do_ln(x):
                # do layer normalization excluding the mask
                x_masked = x * broadcast_mask
                N = tf.reduce_sum(mask_float) * lm_dim
                mean = tf.reduce_sum(x_masked) / N
                variance = tf.reduce_sum(((x_masked - mean) * broadcast_mask) ** 2
                                         ) / N
                return tf.nn.batch_normalization(
                    x, mean, variance, None, None, 1E-12
                )

            # normalize the weights
            normed_weights = tf.split(
                tf.nn.softmax(self.W + 1.0 / n_lm_layers), n_lm_layers
            )
            # split LM layers
            layers = tf.split(lm_embeddings, n_lm_layers, axis=1)

            # compute the weighted, normalized LM activations
            pieces = []
            for w, t in zip(normed_weights, layers):
                if self.do_ln:
                    pieces.append(w * _do_ln(tf.squeeze(t, squeeze_dims=1)))
                else:
                    pieces.append(w * tf.squeeze(t, squeeze_dims=1))
            sum_pieces = tf.add_n(pieces)

            weighted_lm_layers = sum_pieces * self.gamma

        return weighted_lm_layers

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], self.embedding_dim


if __name__ == '__main__':
    import os
    datadir = os.path.join('bilm-tf', 'tests', 'fixtures', 'model')
    vocab_file = os.path.join(datadir, 'vocab_test.txt')
    options_file = os.path.join(datadir, 'options.json')
    weight_file = os.path.join(datadir, 'lm_weights.hdf5')

    from keras.layers import Input, GlobalMaxPooling1D, Dense
    from keras.models import Model

    inp = Input((None, 50), dtype=tf.int32)
    elmo = ELMoEmbedding(options_file, weight_file, 32)(inp)
    layer = GlobalMaxPooling1D()(elmo)
    pre = Dense(1)(layer)

    model = Model(inputs=inp, outputs=pre)
    model.compile(loss='binary_crossentropy', optimizer='adam')
    model.summary()

    raw_context = [
        'Pretrained biLMs compute representations useful for NLP tasks .',
        'They give state of the art performance for many tasks .'
    ]

    tokenized_context = [sentence.split() for sentence in raw_context]

    from bilm import Batcher

    batcher = Batcher(vocab_file, 50)

    context_ids = batcher.batch_sentences(tokenized_context)

    model.fit(x=context_ids, y=np.array([1,0]), batch_size=1)


