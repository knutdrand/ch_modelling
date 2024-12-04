import plotly.express as px
from typing import Tuple
import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt
from flax.training import train_state
import optax
from ch_modelling.models.flax_models.distribution_head import NBHead
from ch_modelling.models.flax_models.outlier_model import OutlierRNN
from ch_modelling.models.flax_models.trainer import Trainer, l2_regularization
import numpy as np
import pandas as pd


df = pd.read_csv('/home/knut/Downloads/data-2.csv', header=1)

df = df[df.columns[2:]]
df['period'] = np.arange(len(df)) % 12
df.ffill(inplace=True)
array = df.values

print(array.shape)
trainer = Trainer(OutlierRNN())

class TrainState(train_state.TrainState):
    key: jax.Array

l2_c = 0.000

def loss_fn(eta, y):
    return -NBHead(eta[..., :2]).log_prob(y[..., 0]).mean() - NBHead(eta[..., 2:]).log_prob(y[..., 1]).mean()

class DataLoader:
    context_lengths = 6
    def __init__(self, data, batch_size):
        self.data = data
        self.idx = 0
        self.batch_size = batch_size
        self.mu = np.mean(data, axis=0)
        self.sigma = np.std(data, axis=0)

    def all(self):
        indices = np.arange(len(self.data)-self.context_lengths-1)
        x = (np.array([self.data[i:i+self.context_lengths] for i in indices])-self.mu)/self.sigma
        y = self.data[indices+self.context_lengths+1]
        return x, y
    
    def __iter__(self):
        for _ in range((len(self.data))//self.batch_size):
            indices = np.random.randint(0, len(self.data)-self.context_lengths-1, self.batch_size)
            x = (np.array([self.data[i:i+self.context_lengths] for i in indices])-self.mu)/self.sigma
            y = self.data[indices+self.context_lengths+1]
            yield x, y


@jax.jit
def train_step(state: TrainState, dropout_key, x, y) -> Tuple[TrainState, jnp.ndarray]:
    dropout_train_key = jax.random.fold_in(key=dropout_key, data=state.step)

    def loss_func(params):
        eta = state.apply_fn(params, x, training=True, rngs={'dropout': dropout_train_key})
        return loss_fn(eta, y) + l2_regularization(params, l2_c)

    grad_func = jax.value_and_grad(loss_func)
    loss, grad = grad_func(state.params)
    state = state.apply_gradients(grads=grad)
    return state, loss

loader = DataLoader(array, 16)
model = OutlierRNN()
x,y = next(iter(loader))
params = model.init(jax.random.PRNGKey(0), x, training=False)    
training_state = TrainState.create(
            apply_fn=model.apply,
            params=params,
            tx=optax.adam(0.001),
            key=jax.random.PRNGKey(40)
        )
dropout_key = jax.random.PRNGKey(40)
for i in range( 500):
    total_loss = 0
    for x, y in iter(loader):
        training_state, cur_loss = train_step(training_state, dropout_key, x, y)
        total_loss += cur_loss
    print(total_loss)
x, y = loader.all()
print(x.shape, y.shape)
eta = model.apply(training_state.params, x, training=False)
#px.histogram(eta[..., 0]).show()
px.line(y).show()
print(eta.shape, y.shape)

for i in range(2):
    head = NBHead(eta[..., i*2:(i*2)+2])
    samples = head.sample(jax.random.PRNGKey(0), (1000,))
    
    pred = np.median(samples, axis=-1)
    local_y = y[..., i]
    q = (samples>local_y[:, None]).mean(axis=-1)
    probs=  head.log_prob(local_y)
    df = pd.DataFrame({'q': q, 'flag': (q>0.9) | (q<0.1), 'value': local_y})
    #px.bar(df, y='q', color='flag').show()
    px.bar(df, y='value', color='flag').show()
    px.line(np.array([local_y, pred]).T).show()
    # df = pd.DataFrame({'pred': pred, 'true':y[..., i]})
    #px.scatter(df, x='pred', y='true').show()

