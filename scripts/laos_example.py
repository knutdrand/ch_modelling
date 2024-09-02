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
    datasset_name = 'full'
    ds = ListDataset(get_dataset(datasset_name), freq='1M')
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
    n_static = 2
    caridinalities = [max(t['feat_static_cat'][i] for t in train_ds) + 1 for i in range(n_static)]
    estimator = DeepAREstimator(
        num_layers=2,
        hidden_size=24,
        dropout_rate=0.3,
        num_feat_static_cat=n_static,
        embedding_dimension=[1] * n_static,
        cardinality=caridinalities,
        prediction_length=custom_ds_metadata["prediction_length"],
        freq='M',
        distr_output=NegativeBinomialOutput(),
        trainer_kwargs={
            "enable_progress_bar": False,
            "enable_model_summary": False,
            "max_epochs": 40,
        },
    )
    metrics = evaluate_on_split(estimator.train(train_ds), test_instances)

    print(metrics)
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
