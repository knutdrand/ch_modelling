
try:
    from jax.scipy import stats
    from jax.random import PRNGKey
    import jax.numpy as jnp
    import jax
    import blackjax
    from jax.scipy.special import expit, logit
    import jax.tree_util as tree_util
except ImportError as e:
    jax, jnp, stats, PRNGKey, blackjax = (None, None, None, None, None)
    expit, logit = (None, None)
    tree_util = None