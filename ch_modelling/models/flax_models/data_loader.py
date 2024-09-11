from typing import Iterable, Tuple

import numpy as np

def nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0]


def interpolate_nans(y):
    y = y.copy()
    for row in y:
        nans, x = nan_helper(row)
        row[nans] = np.interp(x(nans), x(~nans), row[~nans])
    return y


class DataLoader:
    def __init__(self, X, y, forecast_length, context_length=None, do_validation=False):
        self._X = X #n_locations, n_periods, n_features
        self._y = y #n_locations, n_periods
        self._interpolated_y = interpolate_nans(y)
        self._context_length = context_length or X.shape[1] - forecast_length
        self._forecast_length = forecast_length
        self._total_length = self._context_length + forecast_length
        self.validation_mask = np.ones(X.shape[1]-self._total_length+1, dtype=bool)
        if do_validation:
            self._validation_index = (X.shape[1]-self._total_length)//2
            self.validation_mask[self._validation_index-forecast_length:self._validation_index+forecast_length] = False
        self.do_validation = do_validation

    def __iter__(self) -> Iterable[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        starts = np.arange(self._X.shape[1] - self._total_length+1)[self.validation_mask]
        permuted_starts = np.random.permutation(starts)
        return ((self._X[:, start:start+self._total_length],
                 self._interpolated_y[:, start:start+self._context_length],
                 self._y[:, start:start+self._total_length]) for start in permuted_starts)

    def validation_set(self):
        return (self._X[:, self._validation_index:self._validation_index+self._total_length],
                self._interpolated_y[:, self._validation_index:self._validation_index+self._context_length],
                self._y[:, self._validation_index:self._validation_index+self._total_length])

