from climate_health.testing.estimators import  sanity_check_estimator

from ch_modelling.models.flax_models.flax_model import FlaxModel, ProbabilisticFlaxModel, ARModelT


def test_flax_model():
    model = ProbabilisticFlaxModel()
    sanity_check_estimator(model)

def test_ar_model():
    model = ARModelT()
    sanity_check_estimator(model)
