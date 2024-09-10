import pytest
from ..data_fixtures import full_data, train_data, train_data_pop, future_climate_data
from climate_health.external.mlflow import ExternalModel, MlFlowTrainPredictRunner


def test_mlflow_model_train(models_path, train_data_pop, future_climate_data):
    model_directory = 'mlflow_test_project'
    runner = MlFlowTrainPredictRunner(models_path / model_directory)
    model = ExternalModel(runner, working_dir=models_path / model_directory)
    model.train(train_data_pop)

    # raies valueerror
    #with pytest.raises(ValueError):
    #model.predict(train_data_pop)
