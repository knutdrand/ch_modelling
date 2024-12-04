from flax.linen import SimpleCell

from ch_modelling.models.flax_models.flax_model import ARModelT
from ch_modelling.models.flax_models.flax_model_v1 import ARModelTV1, ARModelTV2
from ch_modelling.models.flax_models.rnn_model import ARModel2, Preprocess


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


def ar_model_monthly_v2():
    model = ARModelTV1()
    module = ARModel2(
        Preprocess(output_dim=2, n_hidden=4, embedding_dim=3, dropout_rate=0.4),
        SimpleCell(features=4),
        SimpleCell(features=4))
    model.set_model(module)
    model.n_iter = 2000
    model.context_length = 12
    model.prediction_length = 3
    model.learning_rate = 5e-5
    model.l2_c = 0.05
    return model


def ar_model_weekly_v2():
    model = ARModelTV1()
    model.n_iter = 1000
    model.context_length = 52
    model.prediction_length = 12
    model.learning_rate = 1e-5
    return model

def ar_model_weekly_v3():
    model = ARModelTV2()
    module = ARModel2(
        Preprocess(output_dim=2, dropout_rate=0.3),
        SimpleCell(features=4),
        SimpleCell(features=4))
    model.set_model(module)
    model.n_iter = 1000
    model.context_length = 52
    model.prediction_length = 12
    model.learning_rate = 1e-5
    model.l2_c = 0.05
    return model
