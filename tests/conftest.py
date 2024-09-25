import pytest
from chap_core.assessment.dataset_splitting import train_test_split
from chap_core.data.datasets import ISIMIP_dengue_harmonized


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

