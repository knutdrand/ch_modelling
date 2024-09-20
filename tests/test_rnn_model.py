import jax.numpy as jnp
import pytest
import flax.linen as nn
from ch_modelling.models.flax_models.rnn_model import ARModel, ARModel2, Preprocess, MultiValueARAdder
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
    check_module(ar_module, X, y)


def check_module(ar_module, X, y):
    params = ar_module.init(jax.random.PRNGKey(0), X, y)
    eta = ar_module.apply(params, X, y)
    assert eta.shape == (2, 4, 2)


def test_armodel2(X, y):
    ar2_module = ARModel2(Preprocess(n_locations=2, output_dim=2, dropout_rate=0.2),
                          nn.SimpleCell(features=4),
                          nn.SimpleCell(features=4))
    check_module(ar2_module, X, y)

def test_multivalue_model(X, y):
    ar2_module = ARModel2(Preprocess(n_locations=2, output_dim=2, dropout_rate=0.2),
                          nn.SimpleCell(features=4),
                          nn.SimpleCell(features=4),
                          ar_adder=MultiValueARAdder())
    check_module(ar2_module, X, y)
