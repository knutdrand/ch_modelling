from climate_health.data.datasets import ISIMIP_dengue_harmonized
from climate_health.testing.estimators import sanity_check_estimator

from ch_modelling.models.flax_models.flax_model import FlaxModel, ProbabilisticFlaxModel, ARModelT, MultiCountryModel


def test_flax_model():
    model = ProbabilisticFlaxModel()
    sanity_check_estimator(model)


def test_ar_model():
    model = ARModelT(n_iter=10)
    sanity_check_estimator(model)


def sanity_check_multiestimator(estimator):
    prediction_length = 3
    dataset = [ISIMIP_dengue_harmonized['vietnam'], ISIMIP_dengue_harmonized['brazil']]
    #train, test_generator = train_test_generator(dataset, prediction_length, n_test_sets=1)
    estimator.multi_train(dataset)


def test_multi_train():
    model = MultiCountryModel(n_iter=10)
    sanity_check_multiestimator(model)
    # historic, future, _ = next(test_generator)
    # predictor = estimator.train(train)
    # samples = predictor.predict(historic, future)
    # assert isinstance(samples, DataSet)
    # for location, s in samples.items():
    #     assert len(s) == prediction_length
    #     assert s.samples.shape == (prediction_length, 100)
