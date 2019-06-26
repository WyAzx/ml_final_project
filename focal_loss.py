import tensorflow as tf
from keras import backend as K

'''
Compatible with tensorflow backend
'''


def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.sum(
            (1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

    return focal_loss_fixed


def binary_crossentropy_with_ranking(y_true, y_pred):
    """ Trying to combine ranking loss with numeric precision"""
    # first get the log loss like normal
    logloss = K.mean(K.binary_crossentropy(y_pred, y_true), axis=-1)

    # next, build a rank loss

    # clip the probabilities to keep stability
    y_pred_clipped = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())

    # translate into the raw scores before the logit
    y_pred_score = K.log(y_pred_clipped / (1 - y_pred_clipped))

    # determine what the maximum score for a zero outcome is
    y_pos = K.cast(K.greater_equal(y_true, 0.5), 'float32')
    y_neg = K.cast(K.less(y_true, 0.5), 'float32')

    y_pred_score_zerooutcome_max = K.max(y_pred_score * y_neg)

    # determine how much each score is above or below it
    rankloss = y_pred_score - y_pred_score_zerooutcome_max

    # only keep losses for positive outcomes
    rankloss = rankloss * y_pos

    # only keep losses where the score is below the max
    rankloss = K.square(K.clip(rankloss, -10, 0))

    # average the loss for just the positive outcomes
    rankloss = K.sum(rankloss, axis=-1) / (K.sum(y_pos) + 1)

    # return (rankloss + 1) * logloss - an alternative to try
    return rankloss + logloss
