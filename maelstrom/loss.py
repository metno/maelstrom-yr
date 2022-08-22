import tensorflow as tf
import tensorflow.keras.backend as K

import maelstrom


def get(config, quantiles=None):
    if config["type"] == "quantile_score":
        loss = lambda x, y: maelstrom.loss.quantile_score(x, y, quantiles)
    elif config["type"] == "quantile_score_prob":
        loss = lambda x, y: maelstrom.loss.quantile_score_prob(x, y, quantiles)
    elif config["type"] == "mae":
        loss = maelstrom.loss.mae
    elif config["type"] == "mae_prob":
        loss = maelstrom.loss.mae_prob
    else:
        raise NotImplementedError()

    return loss


def quantile_score(y_true, y_pred, quantiles):
    qtloss = 0
    for i, quantile in enumerate(quantiles):
        err = y_true[..., 0] - y_pred[..., i]
        qtloss += (quantile - tf.cast((err < 0), tf.float32)) * err
    return K.mean(qtloss) / len(quantiles)


def quantile_score_prob(y_true, y_pred, quantiles):
    """

    Weighted version:
    return K.mean((qtloss0 + qtloss1 + qtloss2) / (1 + y_true_std))
    """
    y_true_mean = y_true[..., 0]
    y_true_std = y_true[..., 1]

    d = y_true_std * 1.28155  # scipy.stats.norm.ppf(0.1)

    qtloss = 0
    for i, quantile in enumerate(quantiles):
        err = y_true_mean - y_pred[:, :, :, i]
        curr = (quantiles[i] - tf.cast((err < 0), tf.float32)) * err
        qtloss += curr / (1 + y_true_std)
    return K.mean(qtloss / len(quantiles))


def mae(y_true, y_pred):
    return K.mean(K.abs(y_true[..., 0] - y_pred[..., 0]))


def mae_prob(y_true, y_pred):
    y_true_mean = y_true[..., 0]
    y_true_std = y_true[..., 1]
    diff = K.abs(y_true_mean - y_pred[..., 0])
    return K.mean(diff + 0.8 * y_true_std * K.exp(-1.4 / y_true_std * diff))


def meanfcst(y_true, y_pred):
    return K.mean(y_pred)
    # num_leadtimes = y_true.shape[3]
    # # return K.mean(y_pred[:, :, :, 0:num_leadtimes])


def meanobs(y_true, y_pred):
    return K.mean(y_true)
