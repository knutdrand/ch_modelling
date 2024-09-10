import jax.numpy as jnp
import pytest

from ch_modelling.models.flax_models.rnn_model import ARModel
import jax


@pytest.fixture
def X():
    return jnp.empty((2, 5, 4))

@pytest.fixture()
def y():
    return jnp.empty((2, 3))

@pytest.fixture()
def ar_module():
    return ARModel(n_locations=2)

def test_armodel(ar_module, X, y):
    params = ar_module.init(jax.random.PRNGKey(0), X, y)
    ar_module.apply(params, X, y)
