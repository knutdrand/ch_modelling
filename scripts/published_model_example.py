from chap_core.assessment.prediction_evaluator import evaluate_model
from chap_core.climate_predictor import QuickForecastFetcher
from chap_core.datatypes import FullData
from chap_core.external.external_model import (
    get_model_from_directory_or_github_url,
)
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet

model = get_model_from_directory_or_github_url(
    '/home/knut/Sources/ch_modelling/models/ar_model')

dataset = DataSet.from_csv('/home/knut/Data/ch_data/weekly_laos_data.csv', FullData)
evaluate_model(model, dataset, prediction_length=12, n_test_sets=1,
               report_filename='published_report.pdf',
               weather_provider=QuickForecastFetcher)
