class Function(object):
    def sample(self, n):
        raise NotImplementedError

    def __call__(self, S):
        raise NotImplementedError

    @property
    def parameters(self):
        raise NotImplementedError

    def gradient(self, S):
        """Gradients of the log-likelihood wrt the parameters."""
        raise NotImplementedError

    def project_parameters(self):
        raise NotImplementedError

