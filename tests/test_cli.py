import pytest
from climate_health.datatypes import Samples
from climate_health.spatio_temporal_data.temporal_dataclass import DataSet

from ch_modelling.cli import train, predict
from ch_modelling.model import CHAPEstimator


@pytest.fixture
def training_data_filename(tmp_path, split_dataset):
    path = tmp_path / 'training_data.csv'
    split_dataset[0].to_csv(path)
    return path


@pytest.fixture
def test_data_filename(tmp_path, split_dataset):
    path = tmp_path / 'test_data.csv'
    split_dataset[1].to_csv(path)
    return path


@pytest.fixture
def model_filename(tmp_path, split_dataset):
    path = tmp_path / 'model'
    path.mkdir()
    CHAPEstimator(n_epochs=1).train(split_dataset[0]).save(path)
    return path


@pytest.fixture
def output_filename(tmp_path):
    path = tmp_path / 'predictions.csv'
    return path


def test_train(training_data_filename, model_filename):
    train(training_data_filename, model_filename)
    assert (model_filename / 'gluonts-config.json').exists()


def test_predict(training_data_filename, test_data_filename, model_filename, output_filename):
    predict(model_filename,
            training_data_filename, test_data_filename,
            output_filename)
    assert output_filename.exists()
    assert len(output_filename.read_text().splitlines()) > 1
    samples = DataSet.from_csv(output_filename, Samples)
    assert next(iter(samples.values())).samples.shape[-1] == 100

