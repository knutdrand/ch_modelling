from chap_core.assessment.dataset_splitting import train_test_generator
from chap_core.assessment.prediction_evaluator import evaluate_model
from chap_core.data import datasets
import warnings

import flax.linen as nn

from ch_modelling.models.flax_models.flax_model import ARModelT, MultiARModelT
from ch_modelling.models.flax_models.rnn_model import ARModel2, MultiValueARAdder, Preprocess

warnings.filterwarnings("ignore")


def get_model(n_hidden: int, n_locations: int):
    return ARModel2(Preprocess(n_locations=n_locations, output_dim=2, dropout_rate=0.2),
                    nn.SimpleCell(features=n_hidden),
                    nn.SimpleCell(features=n_hidden),
                    ar_adder=MultiValueARAdder())


def hyperparameter():
    data = datasets.ISIMIP_dengue_harmonized['vietnam']
    train_data, _ = train_test_generator(data, 6, n_test_sets=7)
    n_locations = len(data.locations())
    all_results = {}
    for n_hidden in [4, 8, 12]:
        print(f"Running with {n_hidden} hidden units")
        estimator = MultiARModelT(n_iter=200)
        estimator.set_model(get_model(n_hidden, n_locations))
        results, _ = evaluate_model(estimator, train_data, prediction_length=6, n_test_sets=7)
        all_results[n_hidden] = results['MSE']
    print(all_results)


def training():
    data = datasets.ISIMIP_dengue_harmonized['vietnam']
    train_data, _ = train_test_generator(data, 6, n_test_sets=7)
    estimator = MultiARModelT(n_iter=2000)
    estimator.do_validation = True
    estimator.train(train_data)

if __name__ == '__main__':
    hyperparameter()
    training()

