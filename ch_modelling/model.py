import dataclasses
import json
from pathlib import Path

from chap_core.data import DataSet
from chap_core.data.gluonts_adaptor.dataset import DataSetAdaptor
from chap_core.datatypes import Samples
from chap_core.time_period import PeriodRange
from gluonts.dataset.common import ListDataset
from gluonts.transform import Chain
from gluonts.model import Estimator, Predictor

from ch_modelling.estimators import get_naive_estimator, get_deepar_estimator
from gluonts.time_feature import time_features_from_frequency_str
from gluonts.transform import AddTimeFeatures, AddAgeFeature


@dataclasses.dataclass
class CHAPPredictor:
    gluonts_predictor: Predictor
    prediction_length: int

    def predict(self, history: DataSet, future_data: DataSet, num_samples=100) -> DataSet:
        gluonts_dataset = DataSetAdaptor.to_gluonts_testinstances(history, future_data, self.prediction_length)
        gluonts_dataset = get_transform(self.prediction_length).apply(gluonts_dataset, is_train=False)
        forecasts = self.gluonts_predictor.predict(gluonts_dataset, num_samples=num_samples)
        data = {location: Samples(PeriodRange.from_pandas(forecast.index), forecast.samples.T) for location, forecast in
                zip(history.keys(), forecasts)}
        return DataSet(data)

    def save(self, filename: str):
        filepath = Path(filename)
        filepath.mkdir(exist_ok=True, parents=True)
        self.gluonts_predictor.serialize(filepath)
        open(filepath / 'info.json', 'w').write(f'{{"prediction_length": {self.prediction_length}}}')

    @classmethod
    def load(cls, filename: str):
        prediction_length = json.loads(open(Path(filename) / 'info.json').read())['prediction_length']
        return CHAPPredictor(Predictor.deserialize(Path(filename)),
                             prediction_length)
def get_transform(prediction_length):
    transform = Chain([
        AddTimeFeatures(start_field='start', target_field='target', output_field='time_feat', pred_length=prediction_length, time_features=time_features_from_frequency_str('1M')),
        AddAgeFeature(target_field='target', output_field='age', pred_length=prediction_length)])
    return transform

@dataclasses.dataclass
class CHAPEstimator:
    prediction_length: int = 3
    n_epochs: int = 20

    def train(self, dataset: DataSet) -> CHAPPredictor:
        gluonts_dataset = DataSetAdaptor.to_gluonts(dataset)
        ds = ListDataset(gluonts_dataset, freq="m")
        ds = get_transform(self.prediction_length).apply(ds, is_train=True)

        estimator = get_deepar_estimator(n_locations=len(dataset.keys()),
                                         prediction_length=self.prediction_length,
                                         trainer_kwargs={'max_epochs': self.n_epochs})
        #estimator = get_naive_estimator(dataset, prediction_length=self.prediction_length, n_epochs=self.n_epochs)
        return CHAPPredictor(estimator.train(ds), self.prediction_length)
