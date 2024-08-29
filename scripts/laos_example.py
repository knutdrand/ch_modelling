from climate_health.gluonts_adaptor.dataset import get_split_dataset, get_dataset
from gluonts.time_feature import time_features_from_frequency_str
from gluonts.dataset.common import ListDataset
from gluonts.dataset.split import split
from gluonts.torch import DeepAREstimator
from gluonts.torch.distributions import NegativeBinomialOutput
from gluonts.torch.model.mqf2 import MQF2MultiHorizonEstimator
from gluonts.transform import (AddAgeFeature, AddTimeFeatures, Chain, )
from ch_modelling.evaluation import evaluate_estimator, multi_evaluate_estimator, evaluate_on_split
prediction_length = 6
if __name__ == '__main__':
    custom_ds_metadata = {'prediction_length': prediction_length}
    ds = ListDataset(get_dataset('laos_full_data'), freq='1M')
    transform = Chain([
        AddTimeFeatures(start_field='start', target_field='target', output_field='time_feat', pred_length=prediction_length, time_features=time_features_from_frequency_str('1M')),
        AddAgeFeature(target_field='target', output_field='age', pred_length=prediction_length)])
    #ds = transform.apply(ds, is_train=True)
    #train, test = get_split_dataset('laos_full_data', n_periods=12)
    train_ds, test_template = split(ds, offset=-12)
    test_instances = test_template.generate_instances(prediction_length=prediction_length, windows=7, distance=1)
    #test_instances= (t[0] for t in test_instances)
    # for t in test_instances:
    #     print('###############')
    #     print(len(t[0]['target']))
    #     print(len(t[1]['target']))

    #train_ds = ListDataset(train, freq='1M')
    #test_ds = ListDataset(test, freq='1M')
    estimator = DeepAREstimator(
        num_layers=2,
        hidden_size=24,
        dropout_rate=0.3,
        prediction_length=custom_ds_metadata["prediction_length"],
        freq='M',
        distr_output=NegativeBinomialOutput(),
        trainer_kwargs={
            "enable_progress_bar": False,
            "enable_model_summary": False,
            "max_epochs": 20,
        },
    )
    metrics = evaluate_on_split(estimator.train(train_ds), test_instances)

    print(metrics)
    #print(type(forecast_entry))
    # {'MSE': 35970.76418258945, 'abs_error': 19129.0, 'abs_target_sum': 36529.0, 'abs_target_mean': 174.20952380952383, 'seasonal_error': 99.93248871509414, 'MASE': 1.1130792677355252, 'MAPE': 1.4497787682783034, 'sMAPE': 0.6631322867813564, 'MSIS': 9.023140054304102, 'num_masked_target_values': 17.0, 'QuantileLoss[0.1]': 6158.2, 'Coverage[0.1]': 0.08809523809523809, 'QuantileLoss[0.5]': 19129.0, 'Coverage[0.5]': 0.5095238095238095, 'QuantileLoss[0.9]': 11899.0, 'Coverage[0.9]': 0.9047619047619047, 'RMSE': 189.65960081838583, 'NRMSE': 1.08868675300298, 'ND': 0.5236661282816393, 'wQuantileLoss[0.1]': 0.1685838648744833, 'wQuantileLoss[0.5]': 0.5236661282816393, 'wQuantileLoss[0.9]': 0.3257411919296997, 'mean_absolute_QuantileLoss': 12395.4, 'mean_wQuantileLoss': 0.33933039502860746, 'MAE_Coverage': 0.40476190476190466, 'OWA': nan}
    # {'MSE': 75350.8936954427, 'abs_error': 107517.0, 'abs_target_sum': 172495.0, 'abs_target_mean': 230.75946666666667, 'seasonal_error': 95.65854017433807, 'MASE': 2.236305631161224, 'MAPE': 2.2564974462191265, 'sMAPE': 0.8029869478543599, 'MSIS': 21.181002624920232, 'num_masked_target_values': 52.0, 'QuantileLoss[0.1]': 30249.0, 'Coverage[0.1]': 0.12066666666666664, 'QuantileLoss[0.5]': 107517.0, 'Coverage[0.5]': 0.37386666666666674, 'QuantileLoss[0.9]': 81969.2, 'Coverage[0.9]': 0.8047999999999998, 'RMSE': 274.50117248464113, 'NRMSE': 1.1895554121779957, 'ND': 0.6233050233340097, 'wQuantileLoss[0.1]': 0.17536160468419373, 'wQuantileLoss[0.5]': 0.6233050233340097, 'wQuantileLoss[0.9]': 0.4751975419577379, 'mean_absolute_QuantileLoss': 73245.06666666667, 'mean_wQuantileLoss': 0.42462138999198046, 'MAE_Coverage': 0.3682666666666667, 'OWA': nan}

exit()
# params = {'hidden_size': [8, 16, 32],
#           'dropout_rate': [0.1, 0.2, 0.3]}
# results = {}
# def get_mq_estimator(param_comb):
#     return MQF2MultiHorizonEstimator(
#         num_layers=2,
#         hidden_size=param_comb[0],
#         dropout_rate=param_comb[1],
#         prediction_length=custom_ds_metadata["prediction_length"],
#         freq='M',
#         trainer_kwargs={
#             "enable_progress_bar": False,
#             "enable_model_summary": False,
#             "max_epochs": 20,}
#     )
#
#
# def get_deepar_estimator(param_comb):
#     return DeepAREstimator(
#         num_layers=2,
#         hidden_size=param_comb[0],
#         dropout_rate=param_comb[1],
#         prediction_length=custom_ds_metadata["prediction_length"],
#         freq='M',
#         distr_output=NegativeBinomialOutput(),
#         trainer_kwargs={
#             "enable_progress_bar": False,
#             "enable_model_summary": False,
#             "max_epochs": 20,
#         },
#     )
#
#
# # for param_comb in itertools.product(*params.values()):  # type: ignore
# #     estimator = get_mq_estimator(param_comb)
# #     print(f"Training with params: {param_comb}")
# #     metric = evaluate_estimator(estimator)
# #     results[param_comb] = metric
#
#
#
# print(results)
#
#
#
# #from pathlib import Path
#
# #save_path = Path(__file__).parent.parent / 'models' / f"laos_model"
# #create dir
# #save_path.mkdir(parents=True, exist_ok=True)
# #predictor.serialize(save_path)
#
#
