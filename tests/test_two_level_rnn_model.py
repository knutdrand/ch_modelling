import pytest
import numpy as np
from jax import random
from ch_modelling.models.flax_models.two_level_rnn_model import ImplicitEmbed, TwoLevelRNN, WeatherRNN
import jax.numpy as jnp

@pytest.fixture
def random_input():
    rng = random.PRNGKey(0)
    x = random.normal(rng, (2, 3, 4, 5))  # Example input shape (batch_size, sequence_length, features)
    sequence_lengths = np.array([[4, 3, 2],
                                  [2,3, 4]])  # Example sequence lengths
    y = random.poisson(rng, 0.9, (2, 3))
    return x, sequence_lengths, y


def test_weather_rnn_call(random_input):
    x, sequence_lengths, y = random_input
    model = WeatherRNN()
    variables = model.init(random.PRNGKey(0), x, sequence_lengths)
    output = model.apply(variables, x, sequence_lengths)
    assert output.shape== (2, 3, 5)


def test_two_level_rnn(random_input):
    x, sequence_lengths, y = random_input
    model = TwoLevelRNN(weather_aggregator=WeatherRNN())
    variables = model.init(random.PRNGKey(0), x, y,  sequence_lengths)
    output = model.apply(variables, x, y, sequence_lengths)
    assert output.shape == (2, model.n_periods+y.shape[-1], 2)

def test_embedding(random_input):
    x, sequence_lengths, y = random_input
    model = ImplicitEmbed()
    variables = model.init(random.PRNGKey(0), x)
    output = model.apply(variables, x)
    print(output)
    print(jnp.concatenate([x.mean(axis=-2), output], axis=-1))
    assert output.shape == (2, 3, 5)
    