from pathlib import Path
import numpy as np
import pytest
from chap_core.assessment.dataset_splitting import train_test_split
from chap_core.data.datasets import ISIMIP_dengue_harmonized
from chap_core.datatypes import TimeSeriesData, tsdataclass, FullGEEData
from chap_core.data import DataSet
from chap_core.time_period import PeriodRange, Month

@pytest.fixture
def dataset():
    return ISIMIP_dengue_harmonized['vietnam']

@pytest.fixture
def dataset_brazil():
    return ISIMIP_dengue_harmonized['brazil']


@pytest.fixture
def split_dataset(dataset):
    train, test = train_test_split(dataset, prediction_start_period=dataset.period_range[-3])
    return train, test.remove_field('disease_cases')

@pytest.fixture
def split_dataset_brazil(dataset_brazil):
    train, test = train_test_split(dataset_brazil, prediction_start_period=dataset.period_range[-3])
    return train, test.remove_field('disease_cases')

@tsdataclass
class MixedData(TimeSeriesData):
    disease_cases: int
    temperature: float


@pytest.fixture
def mixed_data():
    return MixedData(time_period=PeriodRange.from_strings(['2020-01', '2020-02', '2020-03']),
        disease_cases=[1, 2, 3], temperature=np.ones((3, 31)))

@pytest.fixture
def bigger_mixed_data():
    np.random.seed(0)
    return MixedData(
        time_period=PeriodRange.from_start_and_n_periods(Month(2020, 1), 24),
        disease_cases=np.random.poisson(10., 24), temperature=np.random.random(24*31).reshape((24, 31))*30)

@pytest.fixture
def future_mix_data():
    n_periods = 3
    return MixedData(
        time_period=PeriodRange.from_start_and_n_periods(Month(2022, 1), n_periods),
        disease_cases=np.arange(n_periods), temperature=np.ones((n_periods, 31)))

@pytest.fixture
def ch_data_path():
    return Path('/home/knut/Data/ch_data/')

@pytest.fixture
def rwanda_data():
    return DataSet.from_pickle('/home/knut/Data/ch_data/rwanda_clean_2020_2024_daily.pkl', FullGEEData)