import numpy as np


def zero_pad(X, seq_len):
    return np.array([x[:seq_len - 1] + [0] * max(seq_len - len(x), 1) for x in X])


def get_vocab_size(X):
    return max([max(x) for x in X]) + 1  # plus the 0th word


def fit_in_vocab(X, vocab_size):
    return [[w for w in x if w < vocab_size] for x in X]
