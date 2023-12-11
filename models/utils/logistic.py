from functools import partial

import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.special import logit
from jax.scipy.stats import logistic as jax_logistic
from jax.typing import ArrayLike

from .distribution import Distribution


class Logistic(Distribution):

    def __init__(self, mu: ArrayLike = 0.0, scale: ArrayLike = 1.0, name: str = None) -> None:
        self._scale = scale
        self.check_params()
        self._mu = mu
        super().__init__(name)

    def check_params(self) -> None:
        assert jnp.all(self._scale > 0), "scale must be positive"

    @partial(jax.jit, static_argnums=(0,))
    def logpdf(self, x: ArrayLike) -> ArrayLike:
        return jax_logistic.logpdf(x, self._mu, self._scale)

    @partial(jax.jit, static_argnums=(0,))
    def pdf(self, x: ArrayLike) -> ArrayLike:
        return jax_logistic.pdf(x, self._mu, self._scale)

    @partial(jax.jit, static_argnums=(0,))
    def cdf(self, x: ArrayLike) -> ArrayLike:
        return jax_logistic.cdf(x, self._mu, self._scale)

    @partial(jax.jit, static_argnums=(0,))
    def cdfinv(self, x: ArrayLike) -> ArrayLike:
        return self._mu + self._scale * logit(x)

    def rvs(self, N: int = 1) -> Array:
        return jax.random.logistic(jax.random.PRNGKey(0), shape=(N,)) * self._scale + self._mu

    def __repr__(self) -> str:
        string = f"Logistic(mu={self._mu}, scale={self._scale}"
        if self._name is not None:
            string += f", name={self._name}"
        return string + ")"
