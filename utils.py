import numpy as np


def bf_gain_cal(cb_learned, ch_set, num_ant):

    cb_r = (1 / np.sqrt(num_ant)) * np.cos(cb_learned)
    cb_i = (1 / np.sqrt(num_ant)) * np.sin(cb_learned)
    ch_r = np.squeeze(ch_set[:, :num_ant])
    ch_i = np.squeeze(ch_set[:, num_ant:])

    bf_gain_1 = np.matmul(cb_r, np.transpose(ch_r))
    bf_gain_2 = np.matmul(cb_i, np.transpose(ch_i))
    bf_gain_3 = np.matmul(cb_r, np.transpose(ch_i))
    bf_gain_4 = np.matmul(cb_i, np.transpose(ch_r))

    bf_gain_r = (bf_gain_1 + bf_gain_2) ** 2
    bf_gain_i = (bf_gain_3 - bf_gain_4) ** 2
    bf_gain_pattern = bf_gain_r + bf_gain_i
    max_gain = np.max(bf_gain_pattern, axis=0)
    bf_gain = np.mean(max_gain)

    return bf_gain


def train_shuffle(X):
    shuffled_idx = np.random.permutation(X.shape[0])
    X_shuffled = X[shuffled_idx,:]
    return X_shuffled
