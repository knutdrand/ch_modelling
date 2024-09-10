import logging
from pathlib import Path

import pandas as pd
import pytest
import yaml
from databricks.sdk.service.serving import ExternalModel

from climate_health.spatio_temporal_data.temporal_dataclass import DataSet
from climate_health.datatypes import ClimateHealthTimeSeries, FullData
from climate_health.testing.external_model import sanity_check_external_model

logging.basicConfig(level=logging.INFO)
from climate_health.external.external_model import (get_model_from_yaml_file, run_command,
                                                    ExternalCommandLineModel,
                                                    get_model_from_directory_or_github_url)
from ..data_fixtures import train_data, train_data_pop, future_climate_data
from climate_health.util import conda_available, docker_available


@pytest.mark.skipif(not conda_available(), reason='requires conda')
def test_r_model_from_folder(models_path, train_data, future_climate_data):
    yaml = models_path / 'testmodel' / 'config.yml'
    model = get_model_from_yaml_file(yaml, working_dir=models_path / 'testmodel')
    model.setup()
    model.train(train_data)
    with pytest.raises(ValueError):
        model.predict(future_climate_data)


def test_python_model_from_folder(models_path, train_data, future_climate_data):
    yaml = models_path / 'naive_python_model' / 'config.yml'
    model = get_model_from_yaml_file(yaml, working_dir=models_path / 'naive_python_model')
    model.train(train_data)
    results = model.predict(future_climate_data)
    assert results is not None


@pytest.mark.skipif(not docker_available(), reason='Requires docker')
def test_python_model_from_folder_with_mlproject_file(models_path):
    path = models_path / 'naive_python_model_with_mlproject_file'
    model = ExternalCommandLineModel.from_mlproject_file(path / 'MLproject')


def test_model_from_string_acceptance(models_path):
    model = get_model_from_directory_or_github_url(models_path / 'naive_python_model_with_mlproject_file')
    model = get_model_from_directory_or_github_url(models_path / 'naive_python_model')
    model = get_model_from_directory_or_github_url("https://github.com/knutdrand/external_rmodel_example.git")


def get_dataset_from_yaml(yaml_path: Path, datatype=ClimateHealthTimeSeries):
    specs = yaml.load(yaml_path.read_text(), Loader=yaml.FullLoader)
    if 'demo_data' in specs:
        path = yaml_path.parent / specs['demo_data']
        df = pd.read_csv(path)
    if 'demo_data_adapter' in specs:
        for to_name, from_name in specs['demo_data_adapter'].items():
            if '{' in from_name:
                new_col = [from_name.format(**df.iloc[i].to_dict()) for i in range(len(df))]
                df[to_name] = new_col
            else:
                df[to_name] = df[from_name]
    # df['disease_cases'] = np.arange(len(df))

    return DataSet.from_pandas(df, datatype)


# @pytest.mark.skipif(not conda_available(), reason='requires conda')
@pytest.mark.parametrize('model_directory', ['ewars_Plus'])
# @pytest.mark.parametrize('model_directory', ['naive_python_model'])
def test_all_external_models_acceptance(model_directory, models_path, train_data_pop, future_climate_data):
    """Only tests that the model can be initiated and that train and predict
    can be called without anything failing"""
    print("Running")
    yaml_path = models_path / model_directory / 'config.yml'
    model = get_model_from_yaml_file(yaml_path, working_dir=models_path / model_directory)
    train_data = get_dataset_from_yaml(yaml_path, FullData)
    model.setup()
    model.train(train_data)
    # results = model.predict(future_climate_data)
    # assert results is not None


# @pytest.mark.skip(reason='Conda is a messs')
@pytest.mark.parametrize('model_directory', ['ewars_Plus'])
def test_external_model_predict(model_directory, models_path):
    yaml_path = models_path / model_directory / 'config.yml'
    train_data = get_dataset_from_yaml(yaml_path, FullData)
    model = get_model_from_yaml_file(yaml_path, working_dir=models_path / model_directory)
    model.setup()
    # model.setup()
    results = model.predict(train_data)
    assert isinstance(results, DataSet)


@pytest.mark.skipif(not conda_available(), reason='requires conda')
def test_run_conda():
    assert conda_available()
    # testing that running command with conda works
    command = "conda --version"
    run_command(command)


def test_run_command():
    command = "echo 'hi'"
    run_command(command)

    with pytest.raises(Exception):
        run_command("this_command_does_not_exist")


def test_get_model_from_github():
    repo_url = "https://github.com/knutdrand/external_rmodel_example.git"
    model = get_model_from_directory_or_github_url(repo_url)
    assert model.name == 'example_model'


def test_get_model_from_local_directory(models_path):
    repo_url = models_path / 'ewars_Plus'
    model = get_model_from_directory_or_github_url(repo_url)
    assert model.name == "ewars_Plus"


def test_external_sanity(models_path):
    sanity_check_external_model(models_path / 'naive_python_model_with_mlproject_file')


def test_external_sanity_deepar(models_path):
    sanity_check_external_model(models_path / 'deepar')

def test_external_sanity_deepar(models_path):
    sanity_check_external_model('https://github.com/sandvelab/chap_auto_ewars')
