from pathlib import Path
import logging
from typing import Optional, Tuple, Iterator

import numpy as np
import pandas as pd

from gluonts.dataset.common import DataEntry, Dataset
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.split import split
from gluonts.dataset.stat import calculate_dataset_statistics
from gluonts.dataset.util import period_index
from gluonts.evaluation import Evaluator
from gluonts.model.forecast import Forecast
from gluonts.model.predictor import Predictor
from gluonts.itertools import maybe_len

from gluonts.evaluation import Evaluator, make_evaluation_predictions
from matplotlib import pyplot as plt

#from scripts.laos_example import train_ds, test_ds


def evaluate_estimator(estimator, train_ds, test_ds):

    predictor = estimator.train(train_ds)
    return evaluate_predictor(predictor, test_ds)

def multi_evaluate_estimator(estimator, train_ds, test_dss):
    predictor = estimator.train(train_ds)
    return evaluate_predictor(predictor, test_dss)

def _to_dataframe(input_label):
    """
    Turn a pair of consecutive (in time) data entries into a dataframe.
    """
    start = input_label[0][FieldName.START]
    targets = [entry[FieldName.TARGET] for entry in input_label]
    full_target = np.concatenate(targets, axis=-1)
    index = period_index(
        {FieldName.START: start, FieldName.TARGET: full_target}
    )
    return pd.DataFrame(full_target.transpose(), index=index)

def evaluate_on_split(predictor, test_instances):
    evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
    forecast_it = predictor.predict(test_instances.input, num_samples=100)
    ts_it = map(_to_dataframe, test_instances)
    forecasts = list(forecast_it)
    tss = list(ts_it)
    agg_metrics, item_metrics = evaluator(tss, forecasts)
    for forecast_entry, ts_entry in zip(forecasts, tss):
        plt.plot(ts_entry[-150:].to_timestamp())
        forecast_entry.plot(show_label=True)
        plt.legend()
        plt.show()
    return agg_metrics

def evaluate_predictor(predictor, test_ds):
    test_ds = list(test_ds)
    evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=test_ds,  # test dataset
        predictor=predictor,  # predictor
        num_samples=100,  # number of sample paths we want for evaluation
    )
    forecasts = list(forecast_it)
    # forecast_entry = forecasts[0]
    tss = list(ts_it)
    print(forecasts[0])
    print(tss[0])
    agg_metrics, item_metrics = evaluator(tss, forecasts)
    print(agg_metrics['QuantileLoss[0.9]'])
    print(len(tss), len(forecasts))
    for forecast_entry, ts_entry in zip(forecasts, tss):
        print(type(forecast_entry))
        plt.plot(ts_entry[-150:].to_timestamp())
        forecast_entry.plot(show_label=True)
        plt.legend()
        plt.show()
    return agg_metrics

