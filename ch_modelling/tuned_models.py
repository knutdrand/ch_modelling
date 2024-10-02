from ch_modelling.models.flax_models.flax_model import ARModelT


def ar_model_weekly_v1():
    model = ARModelT()
    model.n_iter = 1000
    model.context_length = 52
    model.prediction_length = 12
    model.learning_rate = 1e-5
    return model


def ar_model_monthly_v1():
    model = ARModelT()
    model.n_iter = 1000
    model.context_length = 12
    model.prediction_length = 3
    model.learning_rate = 1e-5
    return model
