"""Console script for ch_modelling."""
from climate_health.datatypes import FullData, remove_field
from climate_health.external.external_model import get_model_from_directory_or_github_url
from cyclopts import App
from climate_health.data import DataSet, datasets
from .models.flax_models.flax_model import FlaxModel
from .model import CHAPEstimator, CHAPPredictor
from .registry import registry
from climate_health.assessment.prediction_evaluator import evaluate_model
import warnings
warnings.filterwarnings("ignore")
app = App()


@app.command()
def train(training_data_filename: str, model_path: str, n_epochs: int = 20, prediction_length: int = 3):
    '''
    '''
    dataset = DataSet.from_csv(training_data_filename, FullData)
    predictor = CHAPEstimator(prediction_length, n_epochs).train(dataset)
    predictor.save(model_path)


@app.command()
def predict(model_filename: str, historic_data_filename: str, future_data_filename: str, output_filename: str):
    '''
    This function should just be type hinted with common types,
    and it will run as a command line function
    Simple function
    '''
    dataset = DataSet.from_csv(historic_data_filename, FullData)
    future_data = DataSet.from_csv(future_data_filename, remove_field(FullData, 'disease_cases'))
    predictor = CHAPPredictor.load(model_filename)
    forecasts = predictor.predict(dataset, future_data)
    forecasts.to_csv(output_filename)


@app.command()
def evaluate(model_name: str, country_name: str = 'vietnam', against_soa: bool=False):
    model = registry[model_name]()
    model.prediction_length = 6
    data = datasets.ISIMIP_dengue_harmonized[country_name]
    results, _ = evaluate_model(model, data, prediction_length=6, n_test_sets=7,
                                report_filename=f'test_report_{model_name}_{country_name}.pdf')
    if against_soa:

        model_name = 'https://github.com/sandvelab/chap_auto_ewars'
        soa_model = get_model_from_directory_or_github_url(model_name)
        soa_results, _ = evaluate_model(soa_model, data, prediction_length=6, n_test_sets=7,
                                        report_filename=f'test_report_SOA_{country_name}.pdf')
        print(soa_results)
    print(results)


def main():
    app()


if __name__ == "__main__":
    app()
