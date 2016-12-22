from __future__ import division, print_function

import numpy as np
from scipy.special import expit

class NCE(object):
    def __init__(self, f_model, f_noise):
        self.f_model = f_model
        self.f_noise = f_noise

    def _h(self, data, nu):
        # Implements (5) from
        # http://www.jmlr.org/papers/volume13/gutmann12a/gutmann12a.pdf
        G = self.f_model(data) - self.f_noise(data)
        return expit(G - np.log(nu))

    def _log_h(self, data, nu, mul=1.):
        # Implements (5) from
        # http://www.jmlr.org/papers/volume13/gutmann12a/gutmann12a.pdf
        G = self.f_model(data) - self.f_noise(data)
        return - np.logaddexp(0, mul * (np.log(nu) - G))

    def _objective(self, s_model, s_noise):
        nu = len(s_noise) / len(s_model)
        objective = sum(self._log_h(data, nu) for data in s_model)
        objective += sum(self._log_h(data, nu, -1.) for data in s_noise)
        return objective

    def gradient_sample(self, label, data, nu, out=None):
        assert label in (0, 1)

        grads = self.f_model.gradient(data)  # Model gradient.
        G = self.f_model(data) - self.f_noise(data)  # log-lik difference.
        fact = label - expit(G - np.log(nu))
        for grad in grads:
            grad *= fact
        return grads

    def gradient(self, s_model, s_noise):
        nu = len(s_noise) / len(s_model)

        # initialize things
        grads = [np.zeros_like(param) for param in self.f_model.parameters]

        for i, data in enumerate(chain(s_model, s_noise)):
            label = 1 if i < len(s_model) else 0
            grads_term = self.gradient_sample(label, data, nu)
            for grad, grad_term in zip(grads, grads_term):
                grad += grad_term

        return grads

    def learn(self, s_model, s_noise, n_iter=1000, eta_0=3, compute_LL=False,
              plot=True):
        pars = self.f_model.parameters

        if plot:
            values = []
        for i in range(n_iter):
            eta = eta_0 / np.power(1 + i, 0.9)  # Step size.

            grad = self.gradient(s_model, s_noise)
            for j, p in enumerate(pars):
                p += eta * grad[j] / len(s_model)
            self.f_model.project_parameters()

            if i % 5 == 0 and compute_LL:
                print("      LL ~ %f" % (self.f_model._estimate_LL(s_model)))

            if plot:
                values.append(self._objective(s_model, s_noise))

            self._report(s_model, s_noise, i, grad, eta)

        if plot:
            plt.plot(values, 'bo--')
            plt.show()

    def learn_sgd(self, s_model, s_noise, n_iter=1000, eta_0=1e-1,
                  compute_LL=False, plot=True):
        nu = len(s_noise) / len(s_model)  # Fraction of noise to data samples.
        params = self.f_model.parameters

        data = s_model + s_noise
        labels = [1] * len(s_model) + [0] * len(s_noise)

        if compute_LL:
            print("      LL ~ ", self.f_model._estimate_LL(s_model))

        if plot:
            values = []
        for i in range(n_iter * len(data)):
            idx = np.random.randint(0, len(data))
            eta = eta_0 / np.power(1 + i, 0.1)
            grads = self.gradient_sample(labels[idx], data[idx], nu)
            for param, grad in zip(params, grads):
                param += eta * grad
            self.f_model.project_parameters()

            if i % len(data) == 0:
                pass  # self._report(s_model, s_noise, i, grads, eta)

            if i % (len(data) // 20) == 0 and plot:
                values.append(self._objective(s_model, s_noise))

            if i % (5 * len(data)) == 0 and compute_LL:
                print("      LL ~ %f" % (self.f_model._estimate_LL(s_model)))

            if i % 20 * len(data) == 0:
                pass  # self._checkgradient(s_model, s_noise)

        if plot:
            plt.plot(values, 'ro--')
            plt.show()

    def _report(self, s_model, s_noise, i, grads, eta):
        print("[%3d] obj=%f" % (i, self._objective(s_model, s_noise)))
        print("      ||grad||=%f (eta=%f)" % (
            np.sum([np.linalg.norm(x) for x in grads]), eta))

    def _checkgradient(self, s_model, s_noise, eps=1e-6):
        pars = self.f_model.parameters
        grad_analytic = self.gradient(s_model, s_noise)
        grad_numeric = []
        for p in pars:
            grad_numeric.append(np.zeros_like(p))

        f0 = self._objective(s_model, s_noise)
        for j, p in enumerate(pars):
            for i in range(p.size):
                idx = np.unravel_index(i, p.shape)
                orig = p[idx]
                p[idx] -= eps
                f1 = self._objective(s_model, s_noise)
                p[idx] = orig
                grad_numeric[j][idx] = (f0 - f1) / eps

        print("CHECKING GRADIENT OF NCE.")
        for i in range(len(pars)):
            print("CHECKING PARS #%d" % i)
            print("> analytic")
            print(grad_analytic[i])
            print("> numeric")
            print(grad_numeric[i])

            nrm = np.linalg.norm(grad_analytic[i])
            if nrm < 1e-12:
                nrm = 1
            err = np.linalg.norm(grad_analytic[i] - grad_numeric[i]) / nrm
            print("Error=%f" % err)
            assert err < 1e-2
            print("-" * 30)

        print("*** GRADIENT OK")

