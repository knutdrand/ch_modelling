from .flax_model import NBSkipNaN
import jax


class DistributionHead:
    def __init__(self, eta):
        self._dist = None

    def sample(self, key, shape):
        return self._dist.sample(key, shape)

    def mean(self):
        return self._dist.mean

    def log_prob(self, y):
        return self._dist.log_prob(y)


class NBHead(DistributionHead):
    def __init__(self, eta):
        self._dist = NBSkipNaN(jax.nn.softplus(eta[..., 0]), eta[..., 1])


class MixedHead(DistributionHead):
    def __init__(self, eta):
        self._dist = NBSkipNaN(
            jax.nn.softplus(eta[..., 0]),
            eta[..., 1])

    def sample(self, key, shape):
        return self._dist.sample(key, shape)

    def log_prob(self, y):
        return self._dist.log_prob(y)
