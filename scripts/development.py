from ch_modelling.models.flax_models.flax_model_v1 import ARModelTV1

model = ARModelTV1()
model.n_iter = 1000
model.context_length = 52
model.prediction_length = 12
model.learning_rate = 1e-5
