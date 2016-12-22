from __future__ import division
from __future__ import print_function

import numpy as np
from scipy.special import expit
import numpy.linalg

from functions.function import Function

class DPP(Function):
    def __init__(self, V, K=None):
        self.V = V

        if K is None:
            self.K = np.eye(len(V))
        else:
            assert len(K.shape) == 2
            assert K.shape[0] == K.shape[1]
            self.K = np.copy(K)

    def __call__(self, A):
        assert False  # TODO
        return np.sum(self.s[A]) - self.logz

    @property
    def parameters(self):
        return [self.K]

    def _estimate_LL(self, data):
        ll = 0
        n = self.K.shape[0]
        K_diag = np.diag(self.K)
        diag_idxs = np.diag_indices(n)
        K = np.copy(self.K)
        for sample in data:
            K[diag_idxs] = K_diag - 1.
            K[(diag_idxs[0][sample], diag_idxs[1][sample])] += 1.
            ll += np.log(np.abs(np.linalg.det(K)))

        ll /= len(data)

        return ll

    def _get_proposal_marginal(self, given):
        """
        Computes a proposal for adding an item according to the marginal
        probabiltiy. Returns items sorted by likelihood.
        """
        V = list(self.V)
        candidates = np.delete(np.array(V), given)
        given = set(given)

        K = np.copy(self.K)
        P_given = np.linalg.det(K[np.ix_(list(given), list(given))])

        probs = np.zeros(len(candidates))
        for i, el in enumerate(candidates):
            S = list(given.union(set([el])))

            P_S = np.linalg.det(K[np.ix_(S, S)])

            probs[i] = P_S / P_given

        return candidates[np.argsort(probs)[::-1]]


    def _get_proposal(self, given):
        """
        Computes a proposal for adding an item according to the non-marginal
        probabiltiy. Returns items sorted by likelihood.
        """
        V = list(self.V)
        candidates = np.delete(np.array(V), given)
        given = set(given)

        n = self.K.shape[0]
        K_diag = np.diag(self.K)
        diag_idxs = np.diag_indices(n)
        K = np.copy(self.K)

        probs = np.zeros(len(candidates))
        for i, el in enumerate(candidates):
            S = list(given.union(set([el])))

            K[diag_idxs] = K_diag - 1.
            K[(diag_idxs[0][S], diag_idxs[1][S])] += 1.
            probs[i] = np.abs(np.linalg.det(K))

        return candidates[np.argsort(probs)[::-1]]

