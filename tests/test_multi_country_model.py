import jax
import pytest
from flax.linen import SimpleCell
import jax.numpy as jnp
from ch_modelling.models.flax_models.multi_country_model import MultiCountryModule
from ch_modelling.models.flax_models.rnn_model import Preprocess

@pytest.fixture
def n_locations():
    return [2, 3]

@pytest.fixture
def xs(n_locations):
    return [jnp.empty((n, 5, 4)) for n in n_locations]

@pytest.fixture
def ys(n_locations):
    return [jnp.empty((n, 3)) for n in n_locations]


def test_multi_country_module(xs, ys, n_locations):
    n_countries = 2
    module = MultiCountryModule(
        n_locations=n_locations,
        cell_pre_list=[SimpleCell(features=4) for _ in range(n_countries)],
        cell_post_list=[SimpleCell(features=4) for _ in range(n_countries)]
    )
    params=  module.init(jax.random.PRNGKey(0), xs, ys, training=False)
    eta = module.apply(params, xs, ys, training=False)
