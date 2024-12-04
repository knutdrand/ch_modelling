from flax.nnx import RNN, SimpleCell, Rngs
import jax.numpy as jnp
x= jnp.ones((2, 3, 4, 5))
sequence_lengths = jnp.array([[4, 3, 2],
                              [2, 3, 4]])
RNN(SimpleCell(in_features=5, hidden_features=5, rngs=Rngs(0)))(x, seq_lengths=sequence_lengths, return_carry=True)