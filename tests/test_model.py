from climate_health.assessment.dataset_splitting import train_test_split
from climate_health.data.datasets import ISIMIP_dengue_harmonized
from climate_health.data import DataSet
from ch_modelling.model import CHAPEstimator

import pytest


@pytest.fixture
def dataset():
    return ISIMIP_dengue_harmonized['vietnam']


@pytest.fixture
def predictor(dataset):
    return CHAPEstimator().train(dataset)


def test_train(dataset):
    estimator = CHAPEstimator()
    predictor = estimator.train(dataset)


def test_predict(dataset):
    predictor = CHAPEstimator(n_epochs=2).train(dataset)
    train, test = train_test_split(dataset, prediction_start_period=dataset.period_range[-3])
    forecasts = predictor.predict(train, test.remove_field('disease_cases'))
    assert isinstance(forecasts, DataSet)
