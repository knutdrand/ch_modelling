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

class WeatherAggregator(nn.Module):
    @nn.compact
    def __call__(self, x: Float[Array, 'batch period plen features'], 
                    sequence_lengths: Int[Array, 'batch period'], training=False) -> Float[Array, 'batch period features']:
        return x[..., :28, :].mean(axis=2)


class ConvAggregator(nn.Module):
    features: int
    kernel_size: tuple
    window_size: int
    padding: str
    
    @nn.compact
    def __call__(self, x, sequence_lengths, training=False):
        x = x[..., :28, :]
        x = nn.Conv(features=4, kernel_size=self.kernel_size, padding='VALID')(x)
        x = nn.avg_pool(x, window_shape=(self.window_size, ), padding='VALID')
        x = nn.Conv(features=4, kernel_size=self.kernel_size, padding='VALID')(x)
        x = jax.scipy.special.logsumexp(x, axis=-2)
        return x

class ImplicitEmbed(nn.Module):
    hidden_dim: int = 5
    @nn.compact
    def __call__(self, x: Float[Array, 'batch period plen features']) -> Float[Array, 'batch period features']:
        embed = nn.Embed(num_embeddings=x.shape[0], features=self.hidden_dim)(jnp.arange(x.shape[0]))
        return embed[:, None, :].repeat(x.shape[1], axis=1)

class TwoLevelRNN(nn.Module):
    weather_aggregator: WeatherRNN
    n_periods: int = 7
    hidden_dim: int = 5
    @nn.compact
    def __call__(self,
                  x: Float[Array, 'batch period plen features'],
                  y: Float[Array, 'batch period'],
                  sequence_lengths: Int[Array, 'batch period'],
                  training=False):
        print('x', x.shape, 'y', y.shape)
        
        #embed = nn.Embed(num_embeddings=x.shape[0], features=self.hidden_dim)(jnp.arange(x.shape[0]))
        n_ys = y.shape[1]
        x = x[:, :n_ys, ...]
        sequence_lengths = sequence_lengths[:, :n_ys]
        x = MLP([self.hidden_dim, self.hidden_dim], output_dim=10)(x)
        x = self.weather_aggregator(x, sequence_lengths, training=training)
        embed = ImplicitEmbed()(x)
        x = jnp.concatenate([x, embed], axis=-1)
        x = MLP([self.hidden_dim, self.hidden_dim], output_dim=self.hidden_dim)(x)
        x = jnp.concatenate([x, y[..., None]], axis=-1)
        x = MLP([self.hidden_dim, self.hidden_dim], output_dim=self.hidden_dim)(x)
        carry, states = nn.RNN(nn.SimpleCell(features=self.hidden_dim))(x, return_carry=True)
        new_states = nn.RNN(nn.SimpleCell(features=self.hidden_dim))(x[..., -self.n_periods:,:1], initial_carry=states[..., -1, :])
        states = jnp.concatenate([states, new_states], axis=1)
        x = MLP([self.hidden_dim, 4], output_dim=2)(states)
        return x
        