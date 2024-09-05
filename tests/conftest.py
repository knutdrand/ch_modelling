import pytest
from climate_health.assessment.dataset_splitting import train_test_split
from climate_health.data.datasets import ISIMIP_dengue_harmonized


@pytest.fixture
def dataset():
    return ISIMIP_dengue_harmonized['vietnam']

@pytest.fixture
def split_dataset(dataset):
    train, test = train_test_split(dataset, prediction_start_period=dataset.period_range[-3])
    return train, test.remove_field('disease_cases')

