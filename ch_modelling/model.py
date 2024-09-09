import dataclasses
import json
from pathlib import Path

from climate_health.data import DataSet
from climate_health.data.gluonts_adaptor.dataset import DataSetAdaptor
from climate_health.datatypes import Samples
from climate_health.time_period import PeriodRange
from gluonts.dataset.common import ListDataset
from gluonts.model import Estimator, Predictor

from ch_modelling.estimators import get_naive_estimator, get_deepar_estimator


@dataclasses.dataclass
class CHAPPredictor:
    gluonts_predictor: Predictor
    prediction_length: int

    def predict(self, history: DataSet, future_data: DataSet, num_samples=100) -> DataSet:
        gluonts_dataset = DataSetAdaptor.to_gluonts_testinstances(history, future_data, self.prediction_length)
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


@dataclasses.dataclass
class CHAPEstimator:
    prediction_length: int = 3
    n_epochs: int = 20

    def train(self, dataset: DataSet) -> CHAPPredictor:
        gluonts_dataset = DataSetAdaptor.to_gluonts(dataset)
        ds = ListDataset(gluonts_dataset, freq="m")
        estimator = get_deepar_estimator(n_locations=len(dataset.keys()),
                                         prediction_length=self.prediction_length,
                                         trainer_kwargs={'max_epochs': self.n_epochs})
        #estimator = get_naive_estimator(dataset, prediction_length=self.prediction_length, n_epochs=self.n_epochs)
        return CHAPPredictor(estimator.train(ds), self.prediction_length)
