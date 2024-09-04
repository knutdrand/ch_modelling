from climate_health.data.gluonts_adaptor.dataset import get_dataset
from climate_health.data.datasets import ISIMIP_dengue_harmonized
from climate_health.data import adaptors
from gluonts.time_feature import time_features_from_frequency_str
from gluonts.dataset.common import ListDataset
from gluonts.dataset.split import split
from gluonts.torch import DeepAREstimator
from gluonts.torch.distributions import NegativeBinomialOutput
from gluonts.torch.model.mqf2 import MQF2MultiHorizonEstimator
from gluonts.transform import (AddAgeFeature, AddTimeFeatures, Chain, )
from ch_modelling.evaluation import evaluate_estimator, multi_evaluate_estimator, evaluate_on_split, plot_forecasts
prediction_length = 4
good_params = {'num_layers': 2, 'hidden_size': 24, 'dropout_rate': 0.3, 'num_feat_dynamic_real': 0}
params = {'num_layers': 2, 'hidden_size': 12, 'dropout_rate': 0.2, 'num_feat_dynamic_real': 3}

def main(ds, n_static, prediction_length, metadata=None, name='main', params=params):
    transform = Chain([
        AddTimeFeatures(start_field='start', target_field='target', output_field='time_feat', pred_length=prediction_length, time_features=time_features_from_frequency_str('1M')),
        AddAgeFeature(target_field='target', output_field='age', pred_length=prediction_length)])
    ds = transform.apply(ds, is_train=True)
    #train, test = get_split_dataset('laos_full_data', n_periods=12)
    train_ds, test_template = split(ds, offset=-12)
    test_instances = test_template.generate_instances(prediction_length=prediction_length, windows=12-prediction_length+1, distance=1)
    caridinalities = [max(t['feat_static_cat'][i] for t in train_ds) + 1 for i in range(n_static)]
    estimator = DeepAREstimator(
        **params,
        #num_layers=2,
        #hidden_size=24,
        #dropout_rate=0.3,
        num_feat_static_cat=n_static,
        #num_feat_dynamic_real=3,
        embedding_dimension=[2] * n_static,
        scaling=False,
        cardinality=caridinalities,
        prediction_length=custom_ds_metadata["prediction_length"],
        freq='M',
        distr_output=NegativeBinomialOutput(),
        trainer_kwargs={
            "enable_progress_bar": False,
            "enable_model_summary": False,
            "max_epochs": 2,
        },
    )
    estimator = estimator.train(train_ds) # , validation_data=test_template)
    metrics = evaluate_on_split(estimator, test_instances)
    #dbg = predictor.predict(test_instances.input, num_samples=1)
    plot_forecasts(estimator, test_template.generate_instances(prediction_length=prediction_length, windows=12//prediction_length, distance=prediction_length),
                   metadata, name=name)
    print(metrics)

def remove_predictors(ds):
    return ListDataset([{k: v for k, v in entry.items() if k != 'feat_dynamic_real'} for entry in ds],
                       freq='1M')


if __name__ == '__main__':
    custom_ds_metadata = {'prediction_length': prediction_length}
    # country_name = 'vietnam'
    # ds = ISIMIP_dengue_harmonized
    # ds = ds[country_name]
    # dataset = adaptors.gluonts.from_dataset(ds)
    # metadata = adaptors.gluonts.get_metadata(ds)
    # datasset_name, n_static = country_name, 1
    #
    datasset_name, n_static = 'full', 2
    #country_name = 'laos'
    #datasset_name, n_static = 'laos_full_data', 1
    dataset, metadata = get_dataset(datasset_name, with_metadata=True)
    ds = ListDataset(dataset, freq='1M')
    main(ds, n_static, prediction_length, metadata, f'{datasset_name}_full', params=params)
    #naive_ds = remove_predictors(ds)
    #main(naive_ds, n_static, prediction_length, metadata, name=f'{datasset_name}_naive')

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
