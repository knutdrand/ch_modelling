from chap_core.adaptors.command_line_interface import generate_app

from ch_modelling.models.flax_models.flax_model_v1 import ARModelTV1
import logging
logging.basicConfig(level=logging.INFO)

model = ARModelTV1()
model.n_iter = 1000
model.context_length = 52
model.prediction_length = 12
model.learning_rate = 1e-5

app = generate_app(model)
app()
