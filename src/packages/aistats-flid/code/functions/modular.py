from __future__ import division
from __future__ import print_function

import numpy as np
from scipy.special import expit

from functions.function import Function

class ModularFun(Function):
    def __init__(self, V, s):
        self.V = V
        self.s = s

        self.logz = np.sum(np.log(1 + np.exp(self.s)))

    def __call__(self, A):
        return np.sum(self.s[A]) - self.logz

    @property
    def parameters(self):
        return [self.s]

    def gradient(self, A):
        grad = - expit(self.s)
        grad[A] += 1
        return [grad]

    def project_parameters(self):
        self.logz = np.sum(np.log(1 + np.exp(self.s)))

    def sample(self, n):
        # sample n samples
        probs = expit(self.s).reshape(len(self.V))
        data = []
        for _ in range(n):
            s = np.nonzero(np.random.rand(len(self.V)) <= probs)[0]
            data.append(s.tolist())
        return data

    def _estimate_LL(self, data):
        return sum(self(d) for d in data) / len(data)

