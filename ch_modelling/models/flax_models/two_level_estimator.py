import numpy as np
from ch_modelling.models.flax_models.data_loader import SimpleDataLoader
from ch_modelling.models.flax_models.flax_model_v1 import ARModelTV1, DLDataSet
from ch_modelling.models.flax_models.multilevel_transforms import get_multilevl_x
from ch_modelling.models.flax_models.transforms import get_series
from ch_modelling.models.flax_models.two_level_rnn_model import TwoLevelRNN, WeatherRNN



class TwoLevelEstimator(ARModelTV1):
    @property
    def model(self):
        return TwoLevelRNN(
            weather_rnn=WeatherRNN(hidden_dim=20),
            n_periods=self.prediction_length,
            hidden_dim=20)
    

    def extract_series(self, data):
        return get_multilevl_x(data)

    def _get_dataset(self, data):
        x, y = get_series(data, self.extract_series)
        period_lengths = np.array([period.n_days for period in data.period_range])
        period_lengths=  np.array([period_lengths]*x.shape[0])
        return DLDataSet(x, y, forecast_length=self.prediction_length, context_length=self.context_length, extras=[period_lengths])
    
    def loss_func(self, eta_pred, y_true):
        return -self.distribution_head(eta_pred).log_prob(y_true)
    
    def set_validation_data(self, historic_data, future_data):
        x, y = get_series(historic_data, self.extract_series)
        x = x[:, -self.context_length:]
        y = y[:, -self.context_length:]
        fx, fy = get_series(future_data, self.extract_series)
        full_x = np.concatenate([x, fx], axis=1)
        full_y = np.concatenate([y, fy], axis=1)
        period_lengths = np.array([period.n_days for period in historic_data.period_range[-self.context_length:]])
        period_lengths = np.append(period_lengths, [period.n_days for period in future_data.period_range])
        period_lengths=  np.array([period_lengths]*full_x.shape[0])
        self._validation_loader = SimpleDataLoader(
            DLDataSet(full_x, full_y, forecast_length=self.prediction_length, context_length=self.context_length, extras=[period_lengths]))