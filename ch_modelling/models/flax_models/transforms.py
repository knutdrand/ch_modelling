import dataclasses
from datetime import datetime

import numpy as np
from chap_core.datatypes import FullData
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
import flax.linen as nn
from chap_core.time_period.date_util_wrapper import TimeStamp

def get_x(series):
    year_position = [year_position_from_datetime(period.start_timestamp.date) for period in series.time_period]
    x = np.array((series.rainfall, series.mean_temperature, series.population, year_position)).T
    return x

def get_x_wo_population(series):
    year_position = [year_position_from_datetime(period.start_timestamp.date) for period in series.time_period]
    x = np.array((series.rainfall, series.mean_temperature,  year_position)).T
    return x


def get_series(data: DataSet[FullData], series_extractor=None):
    if series_extractor is None:
        series_extractor = get_x
    xs = []
    ys = []
    for series in data.values():
        x = series_extractor(series)
        xs.append(x)  # type: ignore
        if hasattr(series, 'disease_cases'):
            y = series.disease_cases
            ys.append(y)
    assert not np.any(np.isnan(xs))
    return np.array(xs), np.array(ys)




def get_covid_mask(series):
    period = series.time_period
    mask = (period > TimeStamp.parse('2020-01-01')) & (period < TimeStamp.parse('2022-01-01'))
    return mask


def year_position_from_datetime(dt: datetime) -> float:
    day = dt.timetuple().tm_yday
    return day / 365


def get_feature_normalizer(data_set, i=0):
    x = data_set.predictors(i)
    mu = np.mean(x, axis=(0, 1))
    std = np.std(x, axis=(0, 1))
    return lambda x: x[:i] + ((x[i] - mu) / std,) + x[i + 1:]


@dataclasses.dataclass
class ZScaler:
    mu: np.ndarray
    std: np.ndarray

    def __call__(self, x):
        i = 0
        return x[:i] + ((x[i] - self.mu) / self.std,) + x[i + 1:]

    @classmethod
    def from_data(cls, data_set):
        # all axes except the last
        axes = tuple(range(data_set.predictors(0).ndim - 1))
        return ZScaler(np.mean(data_set.predictors(0), axis=axes),
                        np.std(data_set.predictors(0), axis=axes))


class DataDependentTransform:
    def __init__(self, data_set):
        self.data_set = data_set

    def __call__(self, x) -> tuple:
        ...


class DataDependentNormalizer:
    def __init__(self, data_set):
        self.data_set = data_set

    def __call__(self, x) -> tuple:
        ...


def t_chain(normalizers):
    def t(x):
        for n in normalizers:
            x = n(x)
        return x

    return t
