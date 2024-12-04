import jax
from jaxtyping import Array, Float, Int
import flax
from flax import linen as nn
import jax.numpy as jnp

from ch_modelling.models.flax_models.mlp import MLP

class WeatherRNN(nn.Module):
    output_dim: int = 4
    hidden_dim: int = 5
    @nn.compact
    def __call__(self, x: Float[Array, 'batch period plen features'],
                 sequence_lengths: Int[Array, 'batch period'], training=False) -> Float[Array, 'batch period features']:
        cell = nn.SimpleCell(features=self.hidden_dim)
        batch_shape = x.shape[:2]
        carry, x = nn.RNN(cell)(x.reshape(-1, *x.shape[-2:]), seq_lengths=sequence_lengths.ravel(), return_carry=True)
        return carry.reshape(batch_shape + (-1,))
    
class TwoLevelRNN(nn.Module):
    weather_rnn: WeatherRNN
    n_periods: int = 7
    hidden_dim: int = 5
    @nn.compact
    def __call__(self,
                  x: Float[Array, 'batch period plen features'],
                  y: Float[Array, 'batch period'],
                  sequence_lengths: Int[Array, 'batch period'],
                  training=False):
        
        embed = nn.Embed(num_embeddings=x.shape[0], features=self.hidden_dim)(jnp.arange(x.shape[0]))
        n_ys = y.shape[1]
        x = x[:, :n_ys, ...]
        sequence_lengths = sequence_lengths[:, :n_ys]
        b = x.shape[0]
        x = MLP([self.hidden_dim, self.hidden_dim], output_dim=10)(x)
        x = self.weather_rnn(x, sequence_lengths, training=training)
        #x = x + 
        x = jnp.concatenate([x, embed[:, None, :].repeat(n_ys, axis=1),
                             y[..., None]], axis=-1)
        x = MLP([self.hidden_dim, self.hidden_dim], output_dim=self.hidden_dim)(x)
        carry, states = nn.RNN(nn.SimpleCell(features=self.hidden_dim))(x, return_carry=True)
        new_states = nn.RNN(nn.SimpleCell(features=self.hidden_dim))(jnp.zeros((b, self.n_periods, 1)), initial_carry=carry)
        states = jnp.concatenate([states, new_states], axis=1)
        x = MLP([self.hidden_dim, 4], output_dim=2)(states)
        return x
        