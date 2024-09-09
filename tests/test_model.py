from climate_health.assessment.dataset_splitting import train_test_split
from climate_health.data.datasets import ISIMIP_dengue_harmonized
from climate_health.data import DataSet
from climate_health.adaptors.gluonts import GluonTSEstimator
from ch_modelling.estimators import get_deepar_estimator
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
    estimator = get_deepar_estimator(n_locations=len(dataset.keys()),
                                     prediction_length=3,
                                     trainer_kwargs={'max_epochs': 20})
    model = GluonTSEstimator(estimator)
    a, b = evaluate_model(model,# CHAPEstimator(n_epochs=20),
                          dataset, prediction_length=3, n_test_sets=4, report_filename='test_report.pdf')
    print(a)
    print(b)
