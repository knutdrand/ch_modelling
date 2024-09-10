
from gluonts.torch import DeepAREstimator
from gluonts.torch.distributions import NegativeBinomialOutput


def get_deepar_estimator(n_locations, prediction_length, trainer_kwargs=None):
    return DeepAREstimator(
        num_layers=2,
        hidden_size=8,
        dropout_rate=0.2,
        num_feat_static_cat=1,
        scaling=False,
        embedding_dimension=[2],
        cardinality=[n_locations],
        prediction_length=prediction_length,
        distr_output=NegativeBinomialOutput(),
        freq='M',
        trainer_kwargs=trainer_kwargs)


def get_naive_estimator(dataset, prediction_length, n_epochs=20):
    n_locations = len(dataset.keys())
    good_params = {'num_layers': 2, 'hidden_size': 24, 'dropout_rate': 0.3, 'num_feat_dynamic_real': 0}
    n_static = 1
    estimator = DeepAREstimator(**good_params,
                                # num_layers=2,
                                # hidden_size=24,
                                # dropout_rate=0.3,
                                num_feat_static_cat=n_static,
                                # num_feat_dynamic_real=3,
                                embedding_dimension=[2] * n_static,
                                scaling=False,
                                cardinality=[n_locations],
                                prediction_length=prediction_length,
                                freq='M',
                                distr_output=NegativeBinomialOutput(),
                                trainer_kwargs={
                                    "enable_progress_bar": False,
                                    "enable_model_summary": False,
                                    "max_epochs": n_epochs,
                                })
    return estimator

