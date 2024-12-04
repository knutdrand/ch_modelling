import numpy as np
import pytest
from chap_core.datatypes import TimeSeriesData, tsdataclass
from chap_core.data import DataSet
from chap_core.time_period import PeriodRange
from ch_modelling.models.flax_models.two_level_estimator import TwoLevelEstimator

@pytest.fixture
def mixed_dataset(mixed_data):
    return DataSet({'location': mixed_data})

@pytest.fixture
def bigger_mixed_dataset(bigger_mixed_data):
    return DataSet({'location': bigger_mixed_data})

@pytest.fixture
def future_mix_dataset(future_mix_data):
    return DataSet({'location': future_mix_data})

def test_mixed_data(mixed_dataset): 
    estimator = TwoLevelEstimator()
    estimator.context_length = 1
    estimator.prediction_length = 1
    predictor = estimator.train(mixed_dataset)


def test_mixed_data2(bigger_mixed_dataset, future_mix_dataset): 
    estimator = TwoLevelEstimator()
    estimator.context_length = 2
    estimator.prediction_length = 3
    predictor = estimator.train(bigger_mixed_dataset)
    predictor.predict(bigger_mixed_dataset, future_mix_dataset)
        

