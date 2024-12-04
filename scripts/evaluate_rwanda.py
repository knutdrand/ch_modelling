from chap_core.assessment.prediction_evaluator import plot_forecasts, train_test_generator
from chap_core.predictor.model_registry import registry
from chap_core.data import DataSet
from chap_core.datatypes import FullGEEData, FullData
from chap_core.assessment.prediction_evaluator import evaluate_model
from chap_core.climate_predictor import FetcherNd
import chap_core.model_spec
import pandas as pd
from ch_modelling.models.flax_models.two_level_estimator import TwoLevelEstimator
import logging

from scripts.validation_training import monthly_validated_train

logging.basicConfig(level=logging.INFO)
dataset = DataSet.from_pickle('/home/knut/Data/ch_data/rwanda_clean_2020_2024_daily.pkl', FullGEEData)
model = TwoLevelEstimator()
model.context_length = 12
model.prediction_length = 6
model.l2_c = 0.001
model.n_iter =  1000
model.learning_rate=0.01
#monthly_validated_train(dataset, model, level=1)

model_name = 'two_level_estimator'
file_stem = 'rwanda'
if False:
    model = registry.get_model('chap_ewars_monthly')
    model_name = 'chap_ewars_monthly'
    # def convert(data: FullGEEData):
    #     return FullData(time_period=data.time_period,
    #                     disease_cases=data.disease_cases,
    #                     rainfall=data.total_precipitation_sum,
    #                     population=data.population,
    #                     mean_temperature=data.temperature_2m)
    dataset = DataSet.from_csv('/home/knut/Data/ch_data/rwanda_clean_2020_2024.csv', FullData)

def cheat_evaluate_model(
    estimator,
    data: DataSet,
    prediction_length=3,
    n_test_sets=4,
    report_filename=None,
    weather_provider=None,
):
    """
    Evaluate a model on a dataset on a held out test set, making multiple predictions on the test set
    using the same trained model

    Parameters
    ----------
    estimator : Estimator
        The estimator to train and evaluate
    data : DataSet
        The data to train and evaluate on
    prediction_length : int
        The number of periods to predict ahead
    n_test_sets : int
        The number of test sets to evaluate on

    Returns
    -------
    tuple
        Summary and individual evaluation results
    """

    predictor = estimator.train(data)
    truth_data = {
        location: pd.DataFrame(
            data[location].disease_cases,
            index=data[location].time_period.to_period_index(),
        )
        for location in data.keys()
    }
    if report_filename is not None:
        _, plot_test_generatro = train_test_generator(
            data, prediction_length, n_test_sets, future_weather_provider=weather_provider
        )
        plot_forecasts(predictor, plot_test_generatro, truth_data, report_filename)


results = cheat_evaluate_model(model, dataset, prediction_length=6, n_test_sets=9,
                         report_filename=f'cheating_{file_stem}_{model_name}_report_wo.pdf',
                         weather_provider=FetcherNd)
#print(results)

# ({'MSE': np.float64(2749549.121027407), 'abs_error': np.float64(1239261.0), 'abs_target_sum': np.float64(2927870.0), 'abs_target_mean': np.float64(1807.327160493827), 'seasonal_error': np.float64(2147.2356924687306), 'MASE': np.float64(0.43076742021235837), 'MAPE': np.float64(0.539755277098999), 'sMAPE': np.float64(0.40860563093120195), 'MSIS': np.float64(3.785753429305854), 'num_masked_target_values': np.float64(0.0), 'QuantileLoss[0.1]': np.float64(402118.00000000006), 'Coverage[0.1]': np.float64(0.050617283950617285), 'QuantileLoss[0.5]': np.float64(1239261.0), 'Coverage[0.5]': np.float64(0.39691358024691364), 'QuantileLoss[0.9]': np.float64(959997.4), 'Coverage[0.9]': np.float64(0.8197530864197531), 'RMSE': np.float64(1658.1764444797202), 'NRMSE': np.float64(0.9174744234058025), 'ND': np.float64(0.4232636694935226), 'wQuantileLoss[0.1]': np.float64(0.13734148032528767), 'wQuantileLoss[0.5]': np.float64(0.4232636694935226), 'wQuantileLoss[0.9]': np.float64(0.3278825221065143), 'mean_absolute_QuantileLoss': np.float64(867125.4666666667), 'mean_wQuantileLoss': np.float64(0.2961625573084415), 'MAE_Coverage': np.float64(0.3732510288065843), 'OWA': nan}

#trained = model.train(dataset)
