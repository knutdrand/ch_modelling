from chap_core.assessment.dataset_splitting import train_test_generator
from chap_core.assessment.prediction_evaluator import evaluate_model
from chap_core.climate_predictor import QuickForecastFetcher
from chap_core.datatypes import FullData
from chap_core.external.external_model import get_model_from_directory_or_github_url
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet

from ch_modelling.models.flax_models.flax_model import MultiARModelT, ARModelT, BatchedARModelT

# model_url = '/home/knut/Sources/chap_auto_ewars_weekly'
model = ARModelT()
model.n_iter = 1000
model.context_length = 52
model.prediction_length = 12
model.learning_rate = 1e-5
# model.batch_size = 8

# model = NaiveEstimator()
dataset = DataSet.from_csv('/home/knut/Data/ch_data/weekly_laos_data.csv', FullData)


def validated_train():
    train, test = train_test_generator(dataset, prediction_length=104, n_test_sets=1)
    t = next(test)
    print(t)
    _, _, future_data = t
    model.set_validation_data(train_data[-model.context_length:]  future_data
    model.train(train)


if __name__ == '__main__':
    baseline_model = get_model_from_directory_or_github_url(
        '/home/knut/Sources/chap_auto_ewars_weekly'
    )
    # baseline_results = evaluate_model(baseline_model, dataset, prediction_length=12, n_test_sets=41,
    #                                   report_filename='laos_weekly_report_2_a.pdf',
    #                                   weather_provider=QuickForecastFetcher)
    # print(baseline_results)
    # ({'MSE': 526.2114075426721, 'abs_error': 84950.0, 'abs_target_sum': 181022.0, 'abs_target_mean': 22.44431382429186, 'seasonal_error': 8.686936532901475, 'MASE': 1.159185891516386, 'MAPE': 1.080935569340362, 'sMAPE': 0.5934745057553819, 'MSIS': 9.794233364332166, 'num_masked_target_values': 1785.0, 'QuantileLoss[0.1]': 29531.600000000002, 'Coverage[0.1]': 0.07821400577623126, 'QuantileLoss[0.5]': 84950.0, 'Coverage[0.5]': 0.6010125306611396, 'QuantileLoss[0.9]': 57854.39999999999, 'Coverage[0.9]': 0.955563689604685, 'RMSE': 22.939298322805605, 'NRMSE': 1.0220538931325232, 'ND': 0.4692799770193678, 'wQuantileLoss[0.1]': 0.16313818209941333, 'wQuantileLoss[0.5]': 0.4692799770193678, 'wQuantileLoss[0.9]': 0.31959872280717255, 'mean_absolute_QuantileLoss': 57445.333333333336, 'mean_wQuantileLoss': 0.3173389606419846, 'MAE_Coverage': 0.455563689604685, 'OWA': nan},     item_id         forecast_start  ...  QuantileLoss[0.9]  Coverage[0.9]
    # ({'MSE': 1344.451760795152, 'abs_error': 127585.0, 'abs_target_sum': 167143.0, 'abs_target_mean': 22.687304823444528, 'seasonal_error': 8.680349598321753, 'MASE': 1.8950793780461728, 'MAPE': 2.1057875794495624, 'sMAPE': 0.8556703321704626, 'MSIS': 22.18404129066791, 'num_masked_target_values': 1637.0, 'QuantileLoss[0.1]': 32555.800000000003, 'Coverage[0.1]': 0.22173287496816907, 'QuantileLoss[0.5]': 127585.0, 'Coverage[0.5]': 0.6492219463542993, 'QuantileLoss[0.9]': 98131.2, 'Coverage[0.9]': 0.9101648841354725, 'RMSE': 36.66676643494967, 'NRMSE': 1.6161799173720757, 'ND': 0.7633284074116176, 'wQuantileLoss[0.1]': 0.19477812412126144, 'wQuantileLoss[0.5]': 0.7633284074116176, 'wQuantileLoss[0.9]': 0.5871092417869728, 'mean_absolute_QuantileLoss': 86090.66666666667, 'mean_wQuantileLoss': 0.5150719244399505, 'MAE_Coverage': 0.41016488413547253, 'OWA': nan},     item_id         forecast_start  ...  QuantileLoss[0.9]  Coverage[0.9]

    results = evaluate_model(model, dataset, prediction_length=12, n_test_sets=41, report_filename='laos_weekly_report_2_b.pdf',
                             weather_provider=QuickForecastFetcher)
    print(results)
