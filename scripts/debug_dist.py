import numpy as np
from matplotlib import pyplot as plt

from ch_modelling.models.flax_models.flax_model import NBSkipNaN


def plot_dist():
    d = NBSkipNaN(np.array(10.),
                  np.array(1.))
    n_samples = 100000
    samples = d.sample(None, n_samples)
    counts = np.bincount(samples)
    plt.bar(np.arange(0, len(counts)), counts/n_samples)
    #plt.hist(samples, bins=50)
    #plt.show()
    x = np.arange(0, np.max(samples))
    y = np.exp(d.log_prob(x))
    plt.plot(x, y, label='log_prob', color='red')
    plt.legend()
    plt.show()
plot_dist()
