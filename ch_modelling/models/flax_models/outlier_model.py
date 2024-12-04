from flax import linen as nn

class OutlierRNN(nn.Module):
    @nn.compact
    def __call__(self, x, training=False):
        x = nn.Dense(features=4)(x)
        x = nn.relu(x)
        x, _ = nn.RNN(nn.SimpleCell(features=4), return_carry=True)(x)
        x = nn.Dense(features=4)(x)
        x = nn.relu(x)
        x = nn.Dense(features=4)(x)
        return x