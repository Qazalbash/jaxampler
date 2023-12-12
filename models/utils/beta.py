from functools import partial

import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.stats import beta as jax_beta
from jax.typing import ArrayLike
from tensorflow_probability.substrates import jax as tfp

from .distribution import Distribution


class Beta(Distribution):

    def __init__(self, alpha: ArrayLike, beta: ArrayLike, name: str = None) -> None:
        self._alpha = alpha
        self._beta = beta
        self.check_params()
        super().__init__(name)

    def check_params(self) -> None:
        assert jnp.all(self._alpha > 0), "alpha must be positive"
        assert jnp.all(self._beta > 0), "beta must be positive"

    @partial(jax.jit, static_argnums=(0,))
    def logpdf(self, x: ArrayLike) -> ArrayLike:
        return jax_beta.logpdf(x, self._alpha, self._beta)

    @partial(jax.jit, static_argnums=(0,))
    def pdf(self, x: ArrayLike) -> ArrayLike:
        return jax_beta.pdf(x, self._alpha, self._beta)

    @partial(jax.jit, static_argnums=(0,))
    def logcdf(self, x: ArrayLike) -> ArrayLike:
        return jax_beta.logcdf(x, self._alpha, self._beta)

    @partial(jax.jit, static_argnums=(0,))
    def cdf(self, x: ArrayLike) -> ArrayLike:
        return jax_beta.cdf(x, self._alpha, self._beta)

    @partial(jax.jit, static_argnums=(0,))
    def cdfinv(self, x: ArrayLike) -> ArrayLike:
        return tfp.math.betaincinv(self._alpha, self._beta, x)

    def rvs(self, N: int = 1) -> Array:
        return jax.random.beta(self.get_key(), self._alpha, self._beta, shape=(N,))

    def __repr__(self) -> str:
        string = f"beta(alpha={self._alpha}, beta={self._beta}"
        if self._name is not None:
            string += f", name={self._name}"
        return string + ")"
