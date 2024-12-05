import logging
import plotly.express as px
import numpy as np
import pytest
from chap_core.datatypes import TimeSeriesData, tsdataclass
from chap_core.data import DataSet
from chap_core.time_period import PeriodRange
from ch_modelling.models.flax_models.transforms import ZScaler
from ch_modelling.models.flax_models.two_level_estimator import TwoLevelEstimator
from ch_modelling.models.flax_models.two_level_rnn_model import ConvAggregator, TwoLevelRNN, WeatherAggregator
np.set_printoptions(suppress=True, precision=3)
logging.basicConfig(level=logging.INFO)

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
        
def test_on_rwanda_data(rwanda_data): 
    # rwanda_data = rwanda_data.filter_locations(list(rwanda_data.keys())[:5])
    estimator = TwoLevelEstimator()
    estimator.aggregation = ConvAggregator(features=4, kernel_size=(5,), window_size=5, padding='VALID')
    estimator.n_iter = 1000
    estimator.context_length = 12
    estimator.prediction_length = 6
    estimator.learning_rate = 0.01
    train_data = DataSet({location: data[-30:] for location, data in rwanda_data.items()})
    rwanda_data, historic_data, future_data = single_batch_rwanda_data(rwanda_data, estimator)
    predictor = estimator.train(train_data)
    samples = predictor.predict(historic_data, future_data)
    preds = []
    reals = []
    for location, samples in samples.items():
        medians = np.median(samples.samples, axis=-1)
        preds.append(medians)
        reals.append(future_data[location].disease_cases)
        #px.scatter(x=future_data[location].disease_cases, y=medians, title=location).show()
    px.scatter(x=np.concatenate(reals), y=np.concatenate(preds)).show()

def single_batch_rwanda_data(rwanda_data, estimator):
    pl = estimator.prediction_length
    tl = estimator.context_length+pl
    rwanda_data = DataSet({location: data[-tl:] for location, data in rwanda_data.items()})
    historic_data = DataSet({location: data[-tl:-pl] for location, data in rwanda_data.items()})
    future_data = DataSet({location: data[-pl:] for location, data in rwanda_data.items()})
    return rwanda_data, historic_data, future_data


def test_data_loader(rwanda_data):
    estimator = TwoLevelEstimator()
    estimator.context_length = 6
    estimator.prediction_length = 3
    rwanda_data, historic_data, future_data = single_batch_rwanda_data(rwanda_data, estimator)
    dl = estimator.get_dataset(rwanda_data)
    assert len(dl) == 1
    x, y_ar, y, lens = dl[0]
    print(x, y_ar, y, lens)
    print(x.mean(axis=tuple(range(x.ndim-1))))
    print(x.std(axis=tuple(range(x.ndim-1))))
    assert x.shape[:2] == (30, 9)
    assert y_ar.shape == (30, 6)
    assert y.shape == (30, 9)
    assert lens.shape == (30, 9)
    
    x, ar_y, lens = dl.prediction_instance()
    print(x[0, 0])
    #print(x, ar_y, lens)
    assert x.shape[:2] == (30, 9)
    assert ar_y.shape == (30, 6)
    assert lens.shape == (30, 9)
    assert False
