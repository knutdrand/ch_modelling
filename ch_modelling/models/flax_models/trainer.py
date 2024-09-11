from typing import Iterable, Tuple
from more_itertools import peekable
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state
from matplotlib import pyplot as plt


class TrainState(train_state.TrainState):
    key: jax.Array


def l2_regularization(params, scale=1.0):
    return sum(jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(params) if p.ndim == 2) * scale

DataLoader = Iterable[Tuple[jnp.ndarray,
                            jnp.ndarray,
                            jnp.ndarray]]

class Trainer:
    def __init__(self, model, n_iter=3000):
        self.model = model
        self.n_iter = n_iter

    def train(self, data_loader: DataLoader, loss_fn):
        data_loader = peekable(data_loader)
        x, ar_y, y = data_loader.peek()
        params = self.model.init(jax.random.PRNGKey(0), x, ar_y, training=False)
        dropout_key = jax.random.PRNGKey(40)

        training_state = TrainState.create(
            apply_fn=self.model.apply,
            params=params,
            tx=optax.adam(1e-2),
            key=dropout_key
        )

        @jax.jit
        def train_step(state: TrainState, dropout_key) -> Tuple[TrainState, jnp.ndarray]:
            dropout_train_key = jax.random.fold_in(key=dropout_key, data=state.step)

            def loss_func(params):
                eta = state.apply_fn(params, x, ar_y, training=True, rngs={'dropout': dropout_train_key})
                return loss_fn(eta, y) + l2_regularization(params, 0.001)

            grad_func = jax.value_and_grad(loss_func)
            loss, grad = grad_func(state.params)
            state = state.apply_gradients(grads=grad)
            return state, loss

        for i in range(self.n_iter):
            training_state, cur_loss = train_step(training_state, dropout_key)
            if i % 1000 == 0:
                print(f"Loss: {cur_loss}")

        return training_state
