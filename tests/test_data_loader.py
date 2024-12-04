import jax.numpy as jnp
from chap_core.data import DataSet as FullDataSet
from chap_core.datatypes import FullGEEData
import pytest
import numpy as np

from ch_modelling.models.flax_models.data_loader import MultiDataLoader

n_loc = [2, 3]


@pytest.fixture
def Xs():
    return [np.empty((n_locations, 5, 4)) for n_locations in [2, 3]]


@pytest.fixture
def ys():
    return [np.empty((n, 3)) for n in n_loc]


def test_multi_loader(Xs, ys):
    multi_data_loader = MultiDataLoader(Xs, ys, forecast_length=3, context_length=1)
    epoch = list(multi_data_loader)
    assert len(epoch) == 2
    first_bach = epoch[0]
    assert len(first_bach) == 3
    xs, ar_ys, ys = first_bach
    assert all(x.shape == (n, 4, 4) for x, n in zip(xs, n_loc))
    # assert first_bach[0].shape == (2, 4, 4)
    epoch = list(multi_data_loader)
    first_bach = epoch[0]
    assert len(first_bach) == 3
    xs, ar_ys, ys = first_bach
    assert all(x.shape == (n, 4, 4) for x, n in zip(xs, n_loc))

def test_twolevel_dataloader():
     dataset = FullDataSet.from_pickle('/home/knut/Data/ch_data/rwanda_clean_2020_2024_daily.pkl', FullGEEData)