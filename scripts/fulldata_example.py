from pathlib import Path

from chap_core.gluonts_adaptor.dataset import DataSetAdaptor, get_split_dataset
from chap_core.spatio_temporal_data.multi_country_dataset import MultiCountryDataSet
from gluonts.dataset.common import ListDataset
from gluonts.torch import DeepAREstimator
from gluonts.torch.distributions import NegativeBinomialOutput

from ch_modelling.evaluation import evaluate_estimator

custom_ds_metadata = {'prediction_length': 12}
train, test = get_split_dataset('full', n_periods=custom_ds_metadata['prediction_length'])
train = list(train)
print(train[0])
caridinalities = [max(t['feat_static_cat'][i] for t in train)+1 for i in range(2)]

estimator = DeepAREstimator(
            num_layers=2,
            hidden_size=12,
            weight_decay=1e-4,
            dropout_rate=0.2,
            num_feat_static_cat=2,
            embedding_dimension=[4, 4],
            cardinality=caridinalities,
            prediction_length=custom_ds_metadata["prediction_length"],
            freq='M',
            distr_output=NegativeBinomialOutput(),
            trainer_kwargs={
                "enable_progress_bar": False,
                "enable_model_summary": False,
                "max_epochs": 50,
            },
        )

#191876438.02773434
#72498614112.8277
#1106470.8277343747
#4884897.627734374



train_ds = ListDataset(train, freq='1M')
test_ds = ListDataset(test, freq='1M')

metrics = evaluate_estimator(estimator, train_ds, test_ds)
