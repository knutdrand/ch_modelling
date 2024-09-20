import flax.linen as nn
import jax.numpy as jnp

from ch_modelling.models.flax_models.rnn_model import ARAdder, MLP


class MultiCountryModule(nn.Module):
    n_locations: list
    cell_pre_list: list[nn.RNNCellBase]
    cell_post_list: list[nn.RNNCellBase]
    ar_adder: ARAdder = ARAdder()
    preprocess: nn.Module = MLP([8], 4)
    postprocess: nn.Module = MLP([6], 2)
    output_dim: int = 2

    @nn.compact
    def __call__(self, xs, ys, training=False):
        embeddings = [nn.Embed(num_embeddings=n_locations, features=4)(jnp.arange(n_locations))
                      for n_locations in self.n_locations]
        xs = [self.preprocess(x, training=training)+embedding[:,None,...] for (x, embedding) in zip(xs, embeddings)]
        n_ys = [y.shape[-1] for y in ys]
        prev_xs = [self.ar_adder(x, y) for x, y in zip(xs, ys)]
        states_list = [nn.RNN(cell_pre)(prev_x) for cell_pre, prev_x in zip(self.cell_pre_list, prev_xs)]
        new_states_list = [nn.RNN(cell_post)(x[..., n_y + 1:, :], initial_carry=states[..., -1, :])
                      for cell_post, x, n_y, states in zip(self.cell_post_list, xs, n_ys, states_list)]
        xs = [jnp.concatenate([states, new_states], axis=1) for states, new_states in zip(states_list, new_states_list)]
        xs = [nn.Dense(features=6)(x) for x in xs]
        xs = [nn.relu(x) for x in xs]
        xs = [nn.Dense(features=self.output_dim)(x) for x in xs]
        return xs
