from keras.layers import *
from keras.models import Model
import keras.backend as K
from keras.callbacks import Callback
from keras.optimizers import Adam


class Attention(Layer):
    """多头注意力机制
    """
    def __init__(self, nb_head, size_per_head, **kwargs):
        self.nb_head = nb_head
        self.size_per_head = size_per_head
        self.out_dim = nb_head * size_per_head
        super(Attention, self).__init__(**kwargs)
    def build(self, input_shape):
        q_in_dim = input_shape[0][-1]
        k_in_dim = input_shape[1][-1]
        v_in_dim = input_shape[2][-1]
        self.q_kernel = self.add_weight(name='q_kernel',
                                        shape=(q_in_dim, self.out_dim),
                                        initializer='glorot_normal')
        self.k_kernel = self.add_weight(name='k_kernel',
                                        shape=(k_in_dim, self.out_dim),
                                        initializer='glorot_normal')
        self.v_kernel = self.add_weight(name='w_kernel',
                                        shape=(v_in_dim, self.out_dim),
                                        initializer='glorot_normal')
    def mask(self, x, mask, mode='mul'):
        if mask is None:
            return x
        else:
            for _ in range(K.ndim(x) - K.ndim(mask)):
                mask = K.expand_dims(mask, K.ndim(mask))
            if mode == 'mul':
                return x * mask
            else:
                return x - (1 - mask) * 1e10
    def call(self, inputs):
        q, k, v = inputs[:3]
        v_mask, q_mask = None, None
        if len(inputs) > 3:
            v_mask = inputs[3]
            if len(inputs) > 4:
                q_mask = inputs[4]
        # 线性变化
        qw = K.dot(q, self.q_kernel)
        kw = K.dot(k, self.k_kernel)
        vw = K.dot(v, self.v_kernel)
        # 形状变换
        qw = K.reshape(qw, (-1, K.shape(qw)[1], self.nb_head, self.size_per_head))
        kw = K.reshape(kw, (-1, K.shape(kw)[1], self.nb_head, self.size_per_head))
        vw = K.reshape(vw, (-1, K.shape(vw)[1], self.nb_head, self.size_per_head))
        # 维度置换
        qw = K.permute_dimensions(qw, (0, 2, 1, 3))
        kw = K.permute_dimensions(kw, (0, 2, 1, 3))
        vw = K.permute_dimensions(vw, (0, 2, 1, 3))
        # Attention
        a = K.batch_dot(qw, kw, [3, 3]) / self.size_per_head**0.5
        a = K.permute_dimensions(a, (0, 3, 2, 1))
        a = self.mask(a, v_mask, 'add')
        a = K.permute_dimensions(a, (0, 3, 2, 1))
        a = K.softmax(a)
        # 完成输出
        o = K.batch_dot(a, vw, [3, 2])
        o = K.permute_dimensions(o, (0, 2, 1, 3))
        o = K.reshape(o, (-1, K.shape(o)[1], self.out_dim))
        o = self.mask(o, q_mask, 'mul')
        return o
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.out_dim)


def dilated_gated_conv1d(seq, mask, dilation_rate=1):
    """膨胀门卷积（残差式）
    """
    dim = K.int_shape(seq)[-1]
    h = Conv1D(dim*2, 3, padding='same', dilation_rate=dilation_rate)(seq)
    def _gate(x):
        dropout_rate = 0.1
        s, h = x
        g, h = h[:, :, :dim], h[:, :, dim:]
        g = K.in_train_phase(K.dropout(g, dropout_rate), g)
        g = K.sigmoid(g)
        return g * s + (1 - g) * h
    seq = Lambda(_gate)([seq, h])
    seq = Lambda(lambda x: x[0] * x[1])([seq, mask])
    return seq


def seq_maxpool(x):
    seq, mask = x
    seq -= (1 - mask) * 1e10
    return K.max(seq, 1)


def seq_avgpool(x):
    seq, mask = x
    seq = seq * mask
    return K.mean(seq, 1)


def get_dcnn_model(embedding_matrix, num_aux_targets):
    words = Input(shape=(None,), name='Text-Input')
    mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'), name='Mask')(words)
    emb = Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False)(words)
    # x = SpatialDropout1D(0.2, name='Dropout')(emb)
    x = Dropout(0.5)(emb)
    # x = Bidirectional(CuDNNLSTM(128, return_sequences=True), name='BiLSTM-L1')(x)
    # x = Bidirectional(CuDNNLSTM(128, return_sequences=True), name='BiLSTM-L2')(x)

    t = dilated_gated_conv1d(x, mask, 1)
    t = dilated_gated_conv1d(t, mask, 2)
    t = dilated_gated_conv1d(t, mask, 5)
    t = dilated_gated_conv1d(t, mask, 1)
    t = dilated_gated_conv1d(t, mask, 2)
    t = dilated_gated_conv1d(t, mask, 5)
    t = dilated_gated_conv1d(t, mask, 1)
    t = dilated_gated_conv1d(t, mask, 2)
    t = dilated_gated_conv1d(t, mask, 5)
    t = dilated_gated_conv1d(t, mask, 1)
    t = dilated_gated_conv1d(t, mask, 1)
    t = dilated_gated_conv1d(t, mask, 1)

    hidden = Lambda(seq_maxpool)([t, mask])
    hidden = Dense(256, activation='relu')(hidden)
    result = Dense(1, activation='sigmoid', name='Label-Predict')(hidden)
    aux_result = Dense(num_aux_targets, activation='sigmoid', name='Aux-Label-Predict')(hidden)

    # save_model = Model(inputs=emb, outputs=[result, aux_result])
    model = Model(inputs=[words], outputs=[result, aux_result])

    return model