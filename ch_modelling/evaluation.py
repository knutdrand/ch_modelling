from pathlib import Path

from gluonts.evaluation import Evaluator, make_evaluation_predictions
from matplotlib import pyplot as plt

#from scripts.laos_example import train_ds, test_ds


def evaluate_estimator(estimator, train_ds, test_ds):
    evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
    predictor = estimator.train(train_ds)
    predictor.serialize(Path('./'))
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=test_ds,  # test dataset
        predictor=predictor,  # predictor
        num_samples=100,  # number of sample paths we want for evaluation
    )
    forecasts = list(forecast_it)
    # forecast_entry = forecasts[0]
    tss = list(ts_it)
    agg_metrics, item_metrics = evaluator(tss, forecasts)
    print(agg_metrics['QuantileLoss[0.9]'])
    print(len(tss), len(forecasts))
    for forecast_entry, ts_entry in zip(forecasts, tss):
         plt.plot(ts_entry[-150:].to_timestamp())
         forecast_entry.plot(show_label=True)
         plt.legend()
         plt.show()

    return agg_metrics

