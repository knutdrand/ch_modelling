from climate_health.assessment.dataset_splitting import train_test_split
from climate_health.data.datasets import ISIMIP_dengue_harmonized
from climate_health.data import DataSet
from ch_modelling.model import CHAPEstimator

import pytest





@pytest.fixture
def predictor(dataset):
    return CHAPEstimator().train(dataset)


def test_train(dataset):
    estimator = CHAPEstimator()
    predictor = estimator.train(dataset)


def test_predict(split_dataset):
    train, test = split_dataset
    predictor = CHAPEstimator(n_epochs=2).train(train)
    forecasts = predictor.predict(train, test)
    assert isinstance(forecasts, DataSet)
