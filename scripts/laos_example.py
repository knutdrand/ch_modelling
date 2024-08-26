from climate_health.gluonts_adaptor.dataset import get_split_dataset
from gluonts.dataset.common import ListDataset
from gluonts.torch import DeepAREstimator
from gluonts.torch.distributions import NegativeBinomialOutput
from gluonts.torch.model.mqf2 import MQF2MultiHorizonEstimator

from ch_modelling.evaluation import evaluate_estimator

custom_ds_metadata = {'prediction_length': 12}
train, test = get_split_dataset('laos_full_data', n_periods=custom_ds_metadata['prediction_length'])
train_ds = ListDataset(train, freq='1M')
test_ds = ListDataset(test, freq='1M')


estimator = DeepAREstimator(
            num_layers=2,
            hidden_size=16,
            dropout_rate=0.2,
            prediction_length=custom_ds_metadata["prediction_length"],
            freq='M',
            distr_output=NegativeBinomialOutput(),
            trainer_kwargs={
                "enable_progress_bar": False,
                "enable_model_summary": False,
                "max_epochs": 10,
            },
        )

metrics = evaluate_estimator(estimator, train_ds, test_ds)
exit()
params = {'hidden_size': [8, 16, 32],
          'dropout_rate': [0.1, 0.2, 0.3]}
results = {}
def get_mq_estimator(param_comb):
    return MQF2MultiHorizonEstimator(
        num_layers=2,
        hidden_size=param_comb[0],
        dropout_rate=param_comb[1],
        prediction_length=custom_ds_metadata["prediction_length"],
        freq='M',
        trainer_kwargs={
            "enable_progress_bar": False,
            "enable_model_summary": False,
            "max_epochs": 20,}
    )


def get_deepar_estimator(param_comb):
    return DeepAREstimator(
        num_layers=2,
        hidden_size=param_comb[0],
        dropout_rate=param_comb[1],
        prediction_length=custom_ds_metadata["prediction_length"],
        freq='M',
        distr_output=NegativeBinomialOutput(),
        trainer_kwargs={
            "enable_progress_bar": False,
            "enable_model_summary": False,
            "max_epochs": 20,
        },
    )


# for param_comb in itertools.product(*params.values()):  # type: ignore
#     estimator = get_mq_estimator(param_comb)
#     print(f"Training with params: {param_comb}")
#     metric = evaluate_estimator(estimator)
#     results[param_comb] = metric



print(results)



#from pathlib import Path

#save_path = Path(__file__).parent.parent / 'models' / f"laos_model"
#create dir
#save_path.mkdir(parents=True, exist_ok=True)
#predictor.serialize(save_path)


