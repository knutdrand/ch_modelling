from typing import Tuple, Optional
from more_itertools import peekable
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
import logging
from .data_loader import DataLoader

logger = logging.getLogger(__name__)

class TrainState(train_state.TrainState):
    key: jax.Array


def l2_regularization(params, scale=1.0):
    return sum(jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(params) if p.ndim == 2) * scale


# DataLoader = Iterable[Tuple[jnp.ndarray,
#                            jnp.ndarray,
#                            jnp.ndarray]]

class Trainer:
    def __init__(self, model, n_iter=3000, learning_rate=1e-5, l2_c= 0.001, validation_loader: Optional[DataLoader]=None):
        self.model = model
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self._validation_loader = validation_loader
        self.l2_c = l2_c

    def train(self, data_loader: DataLoader, loss_fn):
        logger.info(f'Starting training with parameters: {self.n_iter}, {self.learning_rate}, {self.l2_c} ')
        ix, iar_y, iy, *extras = peekable(iter(data_loader)).peek()
        print(ix.shape, iar_y.shape, iy.shape, [e.shape for e in extras])
        params = self.model.init(jax.random.PRNGKey(0), ix, iar_y, *extras, training=False)
        dropout_key = jax.random.PRNGKey(40)

        training_state = TrainState.create(
            apply_fn=self.model.apply,
            params=params,
            tx=optax.adam(self.learning_rate),
            key=dropout_key
        )

        @jax.jit
        def train_step(state: TrainState, dropout_key, x, ar_y, y, *extras) -> Tuple[TrainState, jnp.ndarray]:
            dropout_train_key = jax.random.fold_in(key=dropout_key, data=state.step)

            def loss_func(params):
                eta = state.apply_fn(params, x, ar_y, *extras, training=True, rngs={'dropout': dropout_train_key})
                assert eta.shape[1] == y.shape[1], (eta.shape, y.shape)
                return loss_fn(eta, y) + l2_regularization(params, self.l2_c)

            grad_func = jax.value_and_grad(loss_func)
            loss, grad = grad_func(state.params)
            state = state.apply_gradients(grads=grad)
            return state, loss

        @jax.jit
        def get_validation_loss(state: TrainState, x, ar_y, y, *extras):
            return loss_fn(state.apply_fn(state.params, x, ar_y, *extras, training=False), y)

        for i in range(self.n_iter):
            total_loss = 0
            for x, ar_y, y, *extras in iter(data_loader):
                training_state, cur_loss = train_step(training_state, dropout_key, x, ar_y, y, *extras)
                total_loss += cur_loss
            if i % 10 == 0:
                validation_loss = 0
                if self._validation_loader is not None:
                    v_loss = 0
                    for v_x, v_ar, v_y, *extras in iter(self._validation_loader):
                        v_loss += get_validation_loss(training_state, v_x, v_ar, v_y, *extras)
                    validation_loss = v_loss
                    #validation_loss = loss_fn(training_state.apply_fn(training_state.params, v_x, v_ar, training=False),
                    #v_y)
                logger.info(f"Iteration {i}: Loss: {cur_loss}, Validation Loss: {validation_loss}")

        return training_state
