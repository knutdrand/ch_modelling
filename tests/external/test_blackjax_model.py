from functools import partial

import pandas as pd
import pytest

from chap_core.assessment.dataset_splitting import train_test_split_with_weather
from chap_core.assessment.forecast import forecast
from chap_core.datatypes import ClimateHealthTimeSeries
from ch_modelling.models.jax_models.model_spec import SSMForecasterNuts, NutsParams
from ch_modelling.models.jax_models.specs import NaiveSSM, SSMWithoutWeather
from ch_modelling.models.jax_models.regression_model import RegressionModel, HierarchicalRegressionModel
from ch_modelling.models.jax_models.simple_ssm import SSM
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
from chap_core.time_period import Month
from chap_core.time_period.date_util_wrapper import delta_month, Week




def test_blackjax_model_train(blackjax, jax, train_data, model):
    model.train(train_data)


def test_hierarchical_model_train(blackjax, jax, train_data, hierarchical_model):
    hierarchical_model.train(train_data)


@pytest.fixture()
def init_values(jax):
    jnp = jax.numpy
    return {'beta_temp': 0.1, 'beta_lag': 0.1, 'beta_season': jnp.full(12, 0.1)}


@pytest.fixture()
def model(simple_priors, init_values):
    return RegressionModel(simple_priors, init_values, num_warmup=100, num_samples=100)


@pytest.fixture()
def hierarchical_model(simple_priors, init_values):
    return HierarchicalRegressionModel(num_warmup=100, num_samples=100)


@pytest.fixture()
def simple_priors(jax):
    season_prior = partial(jax.scipy.stats.norm.logpdf, 0, 1)
    beta_prior = partial(jax.scipy.stats.norm.logpdf, 0, 1)
    priors = {'beta_temp': beta_prior,
              'beta_lag': beta_prior,
              'beta_season': season_prior}
    return priors


def test_blackjax_model_predict(model, train_data, test_data):
    truth, future_data = test_data
    model.train(train_data)
    model.predict(future_data)


def test_hierarchical_model_predict(hierarchical_model, train_data, test_data):
    truth, future_data = test_data
    hierarchical_model.train(train_data)
    hierarchical_model.predict(future_data)


def test_ssm_train(train_data, test_data, jax, blackjax):
    model = SSM()
    model.train(train_data)


def test_ssmspe_train(train_data, test_data, jax, blackjax):
    spec = NaiveSSM()
    model = SSMForecasterNuts(spec, NutsParams(n_samples=10, n_warmup=10))
    model.train(train_data)


def test_ssmspe_predict(train_data, test_data, jax, blackjax):
    spec = NaiveSSM()
    model = SSMForecasterNuts(spec, NutsParams(n_samples=10, n_warmup=10))
    model.train(train_data)
    model.predict(test_data[1])


def test_ssmspe_summary(train_data, test_data, jax, blackjax):
    spec = NaiveSSM()
    model = SSMForecasterNuts(spec, NutsParams(n_samples=10, n_warmup=10))
    model.train(train_data)
    summary = model.prediction_summary(test_data[1], 10)
    assert isinstance(summary, DataSet)


def test_model_without_weather(health_population_data, jax, blackjax, fast_params, data_path):
    spec = SSMWithoutWeather()
    model = SSMForecasterNuts(spec, fast_params)
    model.train(health_population_data)
    model.save(data_path / 'model_without_weather')
    model.prediction_summary(Week(health_population_data.end_timestamp))


def test_model_without_weather_predict(health_population_data, jax, blackjax, data_path):
    model = SSMForecasterNuts.load(data_path / 'model_without_weather')
    model.prediction_summary(Week(health_population_data.end_timestamp))


def test_ssm_predict(train_data, test_data, jax, blackjax):
    truth, future_data = test_data
    model = SSM()
    model.train(train_data)
    model.predict(future_data)


def test_ssm_summary(trained_model, test_data, jax, blackjax):
    truth, future_data = test_data
    trained_model
    summaries = trained_model.prediction_summary(future_data, 10)
    assert isinstance(summaries, DataSet)


def test_ssm_forecast(trained_model, test_data, jax, blackjax):
    truth, future_data = test_data
    forecasts = trained_model.forecast(future_data, 10, forecast_delta=2 * delta_month)
    for location, forecast in forecasts.items():
        assert len(forecast.data().time_period) == 2


def test_ssm_forecast_plot(data, jax, blackjax):
    data = data.restrict_time_period(slice(None, Month(2016, 1)))
    model = SSM()
    model.n_warmup = 10
    forecast(model, data, 36 * delta_month)


@pytest.fixture()
def trained_model(train_data, jax, blackjax) -> SSM:
    model = SSM()
    model.n_warmup = 10
    model.train(train_data)
    return model
