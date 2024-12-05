import dataclasses
from npstructures import RaggedArray
import numpy as np
from ch_modelling.models.flax_models.transforms import year_position_from_datetime
from chap_core.datatypes import TimeSeriesData
from chap_core.time_period import PeriodRange, delta_day

def get_multilevl_x(series: TimeSeriesData):
    period_lengths = [period.n_days for period in series.time_period]
    day_range = PeriodRange(series.time_period[0].start_timestamp, series.time_period[-1].end_timestamp, time_delta=delta_day)
    year_positions = [year_position_from_datetime(day.start_timestamp.date) for day in day_range]
    year_array = RaggedArray(year_positions, period_lengths).as_padded_matrix()
    field_names = [field.name for field in dataclasses.fields(series)]
    arrays = [getattr(series, attr) for attr in field_names if attr not in ['time_period', 'disease_cases']]+[year_array]
    shape = next(array.shape for array in arrays if array.ndim == 2)
    arrays = [array if array.ndim == 2 else np.repeat(array[:, None], shape[1], axis=1) for array in
                arrays]
    array = np.array(arrays)
    return np.moveaxis(array, 0, -1)


