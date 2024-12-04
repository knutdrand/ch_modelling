import jax
from jaxtyping import Array, Float, Int
import flax
from flax import linen as nn
import jax.numpy as jnp

class MLP(nn.Module):
    hidden_dims: list[int]
    output_dim: int = 1

    @nn.compact
    def __call__(self, x, training=False):
        for i in range(len(self.hidden_dims)):
            x = nn.Dense(features=self.hidden_dims[i])(x)
            x = nn.relu(x)
        return nn.Dense(features=self.output_dim)(x)
