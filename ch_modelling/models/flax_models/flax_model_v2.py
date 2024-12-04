from flax.linen import SimpleCell

from .flax_model_v1 import ARModelTV1
from .rnn_model import RNNModel, ARModel2, Preprocess


class ARModelTV2(ARModelTV1):
    model: RNNModel = ARModel2(
    Preprocess(output_dim=2, dropout_rate=0.2),
    SimpleCell(features=4),
    SimpleCell(features=4),
    future_x_slice=slice(3, None))

