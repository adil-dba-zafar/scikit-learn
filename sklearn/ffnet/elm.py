import numpy as np
from scipy.linalg import pinv2

from ..base import BaseEstimator, ClassifierMixin
from ..preprocessing import LabelBinarizer
from ..utils import check_random_state, safe_asarray
from ..utils.extmath import safe_sparse_dot


class ELMClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_hidden, random_state=None):
        self.n_hidden = n_hidden
        self.random_state = check_random_state(random_state)

    def fit(self, X, y):
        X = safe_asarray(X)
        n_samples, n_features = X.shape

        lbin = LabelBinarizer()
        Y = lbin.fit_transform(y)

        w = self.random_state.randn(n_features, self.n_hidden)
        b = self.random_state.randn(self.n_hidden)

        H = np.tanh(safe_sparse_dot(X, w) + b)
        beta = np.dot(pinv2(H), Y)

        self.classes_ = lbin.classes_
        self.coef1_ = w
        self.bias1_ = b
        self.coef2_ = beta

        return self

    def predict(self, X):
        X = safe_asarray(X)
        H = np.tanh(safe_sparse_dot(X, self.coef1_) + self.bias1_)
        return np.argmax(np.dot(H, self.coef2_), axis=1)
