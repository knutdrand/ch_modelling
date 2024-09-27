import jax
import numpy as np
import pytest
import pandas as pd
from chap_core.datatypes import FullData, HealthData
from ch_modelling.models.jax_models.hierarchical_model import HierarchicalModel, SeasonalClimateHealthData, \
    create_seasonal_data, HierarchicalStateModelD2
from ch_modelling.models.jax_models.model_spec import PoissonSkipNaN
from ch_modelling.models.jax_models.prototype_hierarchical import GlobalSeasonalParams, \
    get_hierarchy_logprob_func, DistrictParams
from ch_modelling.models.jax_models.protoype_annotated_spec import Positive
from ch_modelling.models.jax_models.utii import state_or_param, get_state_transform
from ch_modelling.models.jax_models.jax import jnp
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
from tests.external.util import check_model



def test_train5(random_key, data_path):
    file_name = (data_path / 'hydromet_5_filtered').with_suffix('.csv')
    filtered = pd.read_csv(file_name)
    #filtered: pd.DataFrame = csv.loc[~np.isnan(csv['disease_cases'])]
    #write to file
    # filtered.to_csv((data_path / 'hydromet_5_filtered').with_suffix('.csv'))

    data = DataSet.from_pandas(filtered, FullData)
    model = HierarchicalModel(random_key, {}, num_warmup=200, num_samples=100)
    model.train(data)


@pytest.mark.parametrize('model_class', [HierarchicalStateModelD2])# , HierarchicalModel])
def test_training(full_train_data, random_key, test_data, model_class):
    model = model_class(random_key, {}, num_warmup=50, num_samples=50)
    check_model(full_train_data, model_class, random_key, test_data)

    # for key, value in results.items():
    #     print(key, value.data())
    #     true = true_data.get_location(key).data().disease_cases
    #     print(key, true)
    #     fig = px.line(y=value.data())
    #     fig.add_scatter(y=true)
    #     fig.show()


@state_or_param
class ParamClass(GlobalSeasonalParams):
    observation_rate: Positive = 0.01
    year_effect: np.ndarray = tuple((0.,)) * 13


def ch_regression(params: ParamClass, given: SeasonalClimateHealthData) -> HealthData:
    log_rate = params.alpha + params.beta * given.mean_temperature + params.month_effect[given.month - 1] + \
               params.year_effect[given.year - 2000]
    final_rate = jnp.exp(log_rate) * given.population * params.observation_rate + 0.1
    return PoissonSkipNaN(final_rate)


def dummy_regression(params: ParamClass, given: SeasonalClimateHealthData) -> HealthData:
    log_rate = params.alpha + params.beta * given.mean_temperature + params.month_effect[given.month - 1] + \
               params.year_effect[given.year - 2000]

    class Dummy:
        def log_prob(self, x):
            rate = jnp.exp(log_rate) * given.population * params.observation_rate
            res = jnp.where(jnp.isnan(x), 0, (rate - jnp.where(jnp.isnan(x), 0, x)))
            print(res)
            # res = jnp.where(jnp.isnan(x), 0, jax.scipy.stats.poisson(x, rate) ** 2)
            return jnp.sum(res ** 2)

    return Dummy()


def test_grad(full_train_data, random_key, jax):
    train_data = full_train_data
    n_years = 13
    T_Param, transform, inv = get_state_transform(ParamClass)
    T_ParamD, transformD, invD = get_state_transform(DistrictParams)
    data_dict = {key: create_seasonal_data(value.data()) for key, value in train_data.items()}
    logprob_func = get_hierarchy_logprob_func(ParamClass, DistrictParams, data_dict, dummy_regression,
                                              observed_name='disease_cases')
    init_params = T_Param(0., 0., 0., np.zeros(12), np.log(0.01), np.zeros(n_years)), {name: T_ParamD(0., 0.) for name
                                                                                       in data_dict.keys()}
    assert not np.isnan(logprob_func(init_params)) and not np.isinf(logprob_func(init_params))
    grad = jax.grad(logprob_func)(init_params)
    print('----------------')
    print(grad)
    assert not np.isnan(grad[0].alpha)
    # assert not np.isnan(jax.tree_util.tree_flatten(grad)[0]).any() and not np.isinf(jax.tree_util.tree_flatten(grad)[0]).any()


def test_poisson_nan():
    observed = jnp.array([1., 2., 3., np.nan])
    rate = jnp.array(4.)

    def log_prob(param):
        return jnp.sum(jnp.where(jnp.isnan(observed),
                                 0,
                                 (param - jnp.where(jnp.isnan(observed), 0, observed) ** 2)))

    def log_prob2(param):
        return jnp.sum((param - observed[~jnp.isnan(observed)]) ** 2)

    assert not np.isnan(log_prob2(rate)), log_prob2(rate)
    assert not np.isnan(jax.grad(log_prob2)(rate)), jax.grad(log_prob2)(rate)
    grad = jax.grad(log_prob)(rate)
    assert not np.isnan(grad).any()
