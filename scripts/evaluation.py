from pathlib import Path
import logging
import cyclopts
from ch_modelling import tuned_models
from chap_core.assessment.dataset_splitting import train_test_generator
from chap_core.assessment.prediction_evaluator import evaluate_model
from chap_core.climate_predictor import QuickForecastFetcher
from chap_core.datatypes import ClimateHealthData, FullData
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet

from ch_modelling.tuned_models import ar_model_weekly_v1, ar_model_monthly_v1, ar_model_monthly_v2, ar_model_weekly_v3

logging.basicConfig(level=logging.INFO)


app = cyclopts.App()

def monthly_validated_train(dataset, model, level=2):
    train, test = train_test_generator(dataset, prediction_length=12, n_test_sets=1)
    if level == 2:
        train, validation = train_test_generator(train, prediction_length=12, n_test_sets=1)
    else:
        validation = test

    t = next(validation)
    historic_data, _, future_data = t
    model.set_validation_data(historic_data, future_data)
    return model.train(train)


@app.command()
def evaluate(data_filename: Path):
    print(f'Evaluating data from {data_filename}')
    file_stem = data_filename.stem
    model = ar_model_weekly_v3()
    dataset = DataSet.from_csv(data_filename, FullData)
    print(dataset)
    results = evaluate_model(model, dataset, prediction_length=12, n_test_sets=41,
                             report_filename=f'{file_stem}_report_wo.pdf',
                             weather_provider=QuickForecastFetcher)
    print(results)

@app.command()
def validation_train(data_filename: Path, level: int = 2, model_name='ar_model_weekly_v3'):
    print(f'Validating training data from {data_filename}')
    file_stem = data_filename.stem
    model = getattr(tuned_models, model_name)()
    dataset = DataSet.from_csv(data_filename, FullData)
    monthly_validated_train(dataset, model, level=level)
    #model.save(f'{file_stem}_model')

if __name__ == '__main__':
    app()


