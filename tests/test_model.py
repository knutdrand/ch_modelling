from climate_health.assessment.dataset_splitting import train_test_split
from climate_health.data.datasets import ISIMIP_dengue_harmonized
from climate_health.data import DataSet
from ch_modelling.model import CHAPEstimator
from climate_health.assessment.prediction_evaluator import evaluate_model
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


def test_evauate_model(dataset):
    a, b = evaluate_model(CHAPEstimator(n_epochs=2),
                          dataset, prediction_length=3, n_test_sets=4, report_filename='test_report.pdf')
    print(a)
    print(b)
