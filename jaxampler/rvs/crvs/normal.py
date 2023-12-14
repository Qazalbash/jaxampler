from functools import partial

import jax
from jax import Array, jit
from jax import numpy as jnp
from jax.scipy.stats import norm as jax_norm
from jax.typing import ArrayLike

from .continuousrv import ContinuousRV


class Normal(ContinuousRV):

    def __init__(self, mu: ArrayLike = 0.0, sigma: ArrayLike = 1.0, name: str = None) -> None:
        self._mu = mu
        self._sigma = sigma
        self.check_params()
        self._logZ = 0.0
        super().__init__(name)

    def check_params(self) -> None:
        assert jnp.all(self._sigma > 0.0), "All sigma must be greater than 0.0"

    @partial(jit, static_argnums=(0,))
    def logpdf(self, x: ArrayLike) -> ArrayLike:
        return jax_norm.logpdf(x, self._mu, self._sigma)

    @partial(jit, static_argnums=(0,))
    def logcdf(self, x: ArrayLike) -> ArrayLike:
        return jax_norm.logcdf(x, self._mu, self._sigma)

    @partial(jit, static_argnums=(0,))
    def pdf(self, x: ArrayLike) -> ArrayLike:
        return jax_norm.pdf(x, self._mu, self._sigma)

    @partial(jit, static_argnums=(0,))
    def cdf(self, x: ArrayLike) -> ArrayLike:
        return jax_norm.cdf(x, self._mu, self._sigma)

    @partial(jit, static_argnums=(0,))
    def ppf(self, x: ArrayLike) -> ArrayLike:
        return jax_norm.ppf(x, self._mu, self._sigma)

    def rvs(self, N: int) -> Array:
        return jax.random.normal(self.get_key(), shape=(N,)) * self._sigma + self._mu

    def __repr__(self) -> str:
        string = f"Normal(mu={self._mu}, sigma={self._sigma}"
        if self._name is not None:
            string += f", name={self._name}"
        return string + ")"
