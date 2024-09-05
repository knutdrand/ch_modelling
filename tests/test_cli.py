import pytest
from ch_modelling.cli import train

@pytest.fixture
def training_data_filename(tmp_path, split_dataset):
    path = tmp_path / 'training_data.csv'
    split_dataset[0].to_csv(path)
    return path

@pytest.fixture
def model_filename(tmp_path):
    path = tmp_path / 'model'
    #mkdir
    path.mkdir()
    return path


def test_train(training_data_filename, model_filename):
    train(training_data_filename, model_filename)
    assert (model_filename/'gluonts-config.json').exists()


