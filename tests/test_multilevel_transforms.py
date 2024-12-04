import pytest
import numpy as np
from npstructures import RaggedArray
from ch_modelling.models.flax_models.multilevel_transforms import get_multilevl_x
from chap_core.datatypes import TimeSeriesData
from chap_core.time_period import PeriodRange, delta_day
from datetime import datetime, timedelta


def test_get_multilevl_x(mixed_data):
    # Create a mock TimeSeriesData object
    # Call the function
    result = get_multilevl_x(mixed_data)

    # Check the shape of the result
    assert result.shape == (3, 31, 2)

