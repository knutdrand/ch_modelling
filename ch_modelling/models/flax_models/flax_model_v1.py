import pickle
import jax
import jax.numpy as jnp
import numpy as np

from chap_core.datatypes import FullData, Samples
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
from .data_loader import DataSet as DLDataSet, SimpleDataLoader
from .distribution_head import NBHead, DistributionHead
from .flax_model import ProbabilisticFlaxModel
from .rnn_model import model_makers
from .trainer import Trainer
from .transforms import get_feature_normalizer, get_series, ZScaler, get_x, get_covid_mask, get_x_wo_population


class FlaxPredictor:
    distribution_head: type[DistributionHead] = NBHead

    def get_samples(self, eta, n_samples):
        self.rng_key, sample_key = jax.random.split(self.rng_key)
        return self.distribution_head(eta).sample(sample_key, (n_samples,))

    def __init__(self, params, transform, model, prediction_length, context_length, data_extractor=None):
        self.model = model
        self._params = params
        self._transform = transform
        self._data_extractor = data_extractor
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.rng_key = jax.random.PRNGKey(1234)


    def predict(self, historic_data: DataSet, future_data: DataSet, num_samples: int = 100):
        assert list(historic_data.keys()) == list(future_data.keys())
        time_period = next(iter(future_data.values())).time_period
        historic_time_period  = next(iter(historic_data.values())).time_period[-self.context_length:]
        x, _ = get_series(future_data, self._data_extractor)
        prev_values, prev_y = get_series(historic_data, self._data_extractor)
        prev_values = prev_values[:, -self.context_length:]
        prev_y = prev_y[:, -self.context_length:]
        full_x = jnp.concatenate([prev_values, x], axis=1)
        period_lengths = np.array([period.n_days for period in historic_time_period] + [period.n_days for period in time_period])
        period_lengths = np.array([period_lengths]*full_x.shape[0])
        print(period_lengths.shape)
        dataset = DLDataSet(full_x, prev_y, forecast_length=self.prediction_length, context_length=self.context_length, extras=[period_lengths])
        dataset.set_transform(self._transform)
        x, y, *extras = dataset.prediction_instance()
        print('x', x.shape, 'y', y.shape)
        print(extras)
        # full_x, iy = self._transform((full_x, interpolate_nans(prev_y)))
        eta = self.model.apply(self._params, x, y, *extras)  # full_x, iy)
        #n_prev = prev_values.shape[1]
        samples = self.get_samples(
            eta[:, -self.prediction_length:], num_samples)
        
        return DataSet(
            {key: Samples(time_period, s) for key, s in zip(future_data.keys(), samples)}
        )

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump((self._params, self._transform), f)

    @classmethod
    def load(cls, path, *args, **kwargs):
        with open(path, 'rb') as f:
            params, transform = pickle.load(f)
        return cls(params, transform, *args, **kwargs)


class ARModelTV1(ProbabilisticFlaxModel):
    rnn_model_name = 'base'
    prediction_length = 3
    n_iter: int = 1000
    context_length = 24
    do_validation = False
    learning_rate = 1e-4
    l2_c = 0.001
    distribution_head: type[DistributionHead] = NBHead
    _validation_loader = None

    def set_model(self, model):
        self._model = model

    def extract_series(self, data):
        return get_x(data)

    @property
    def model(self):
        if self._model is None:
            self._model = model_makers[self.rnn_model_name](self._n_locations)
        return self._model

    def _get_dataset(self, data: DataSet[FullData]):
        x, y = get_series(data, self.extract_series)
        return DLDataSet(x, y, forecast_length=self.prediction_length, context_length=self.context_length)

    def set_validation_data(self, historic_data: DataSet[FullData], future_data: DataSet[FullData]):
        x, y = get_series(historic_data, self.extract_series)
        x = x[:, -self.context_length:]
        y = y[:, -self.context_length:]
        fx, fy = get_series(future_data, self.extract_series)
        full_x = np.concatenate([x, fx], axis=1)
        full_y = np.concatenate([y, fy], axis=1)
        self._validation_loader = SimpleDataLoader(
            DLDataSet(full_x, full_y, forecast_length=self.prediction_length, context_length=self.context_length))

    def train(self, data: DataSet[FullData]):
        data_set = self._get_dataset(data)
        self._transform = ZScaler.from_data(data_set)
        data_set.set_transform(self._transform)
        self._n_locations = len(data.keys())
        data_loader = SimpleDataLoader(data_set)
        trainer = Trainer(self.model, self.n_iter,
                          learning_rate=self.learning_rate,
                          validation_loader=self._validation_loader, l2_c=self.l2_c)
        state = trainer.train(data_loader, self._loss)
        self._params = state.params
        return FlaxPredictor(self._params, self._transform, self.model, self.prediction_length, self.context_length,
                             data_extractor=self.extract_series)

    def load_predictor(self, path):
        return FlaxPredictor.load(path, self.model, self.prediction_length, self.context_length)

    def predict(self, historic_data: DataSet, future_data: DataSet, num_samples: int = 100):
        assert list(historic_data.keys()) == list(future_data.keys())
        x, _ = get_series(future_data, self.extract_series)
        prev_values, prev_y = get_series(historic_data, self.extract_series)
        prev_values = prev_values[:, -self.context_length:]
        prev_y = prev_y[:, -self.context_length:]
        full_x = jnp.concatenate([prev_values, x], axis=1)
        dataset = DLDataSet(full_x, prev_y, forecast_length=self.prediction_length, context_length=self.context_length)
        dataset.set_transform(self._transform)
        x, y = dataset.prediction_instance()
        # full_x, iy = self._transform((full_x, interpolate_nans(prev_y)))
        eta = self.model.apply(self._params, x, y)  # full_x, iy)
        n_prev = prev_values.shape[1]
        samples = self.get_samples(
            eta[:, n_prev - 1:], num_samples)
        time_period = next(iter(future_data.values())).time_period
        return DataSet(
            {key: Samples(time_period, s) for key, s in zip(future_data.keys(), samples)}
        )

    def loss_func(self, eta_pred, y_true) -> jnp.ndarray:
        return -self.distribution_head(eta_pred).log_prob(y_true[..., 1:])

    def _loss(self, y_pred, y_true):
        L = self.loss_func(y_pred, y_true)
        return jnp.mean(L[:, -self.prediction_length:]) / self.context_length + jnp.mean(L[:, -self.prediction_length:])

    def get_samples(self, eta, n_samples):
        self.rng_key, sample_key = jax.random.split(self.rng_key)
        return self.distribution_head(eta).sample(sample_key, (n_samples,))
        # return self._get_dist(eta).sample(eta, (n_samples,))


class ARModelTV2(ARModelTV1):
    def extract_series(self, data):
        return np.hstack([get_x(data), get_covid_mask(data).reshape(-1, 1)])
