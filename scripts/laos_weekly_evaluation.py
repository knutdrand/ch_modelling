from chap_core.assessment.dataset_splitting import train_test_generator
from chap_core.assessment.prediction_evaluator import evaluate_model
from chap_core.climate_predictor import QuickForecastFetcher
from chap_core.datatypes import FullData
from chap_core.external.external_model import get_model_from_directory_or_github_url
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet

from ch_modelling.models.flax_models.flax_model import MultiARModelT, ARModelT, BatchedARModelT

#model_url = '/home/knut/Sources/chap_auto_ewars_weekly'
model = ARModelT()
model.n_iter = 1000
model.context_length = 52
model.prediction_length = 12
model.learning_rate = 1e-5
# model.batch_size = 8

# model = NaiveEstimator()
dataset = DataSet.from_csv('/home/knut/Data/ch_data/weekly_laos_data.csv', FullData)
def validated_train():
    train, test = train_test_generator(dataset, prediction_length=104, n_test_sets=1)
    t = next(test)
    print(t)
    _, _, future_data = t
    model.set_validation_data(future_data)
    model.train(train)

if __name__ == '__main__':
    evaluate_model(model, dataset, prediction_length=12, n_test_sets=41,
                   report_filename='laos_weekly_report_2_b.pdf',
                   weather_provider=QuickForecastFetcher)
