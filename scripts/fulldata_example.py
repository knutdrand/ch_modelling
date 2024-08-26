from pathlib import Path

from climate_health.gluonts_adaptor.dataset import DataSetAdaptor, get_split_dataset
from climate_health.spatio_temporal_data.multi_country_dataset import MultiCountryDataSet
from gluonts.dataset.common import ListDataset
from gluonts.torch import DeepAREstimator
from gluonts.torch.distributions import NegativeBinomialOutput

from ch_modelling.evaluation import evaluate_estimator

custom_ds_metadata = {'prediction_length': 12}
train, test = get_split_dataset('full', n_periods=custom_ds_metadata['prediction_length'])

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


train_ds = ListDataset(train, freq='1M')
test_ds = ListDataset(test, freq='1M')

metrics = evaluate_estimator(estimator, train_ds, test_ds)
