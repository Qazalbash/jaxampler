from functools import partial

import jax
from jax import Array, jit
from jax import numpy as jnp
from jax.scipy.stats import truncnorm as jax_truncnorm
from jax.typing import ArrayLike

from .continuousrv import ContinuousRV


class TruncNormal(ContinuousRV):

    def __init__(self,
                 mu: ArrayLike,
                 sigma: ArrayLike,
                 low: ArrayLike = 0,
                 high: ArrayLike = 1,
                 name: str = None) -> None:
        self._mu = mu
        self._sigma = sigma
        self._low = low
        self._high = high
        self.check_params()
        self._alpha = (self._low - self._mu) / self._sigma
        self._beta = (self._high - self._mu) / self._sigma
        super().__init__(name)

    def check_params(self) -> None:
        assert jnp.all(self._low < self._high), "low must be smaller than high"
        assert jnp.all(self._sigma > 0), "sigma must be positive"

    @partial(jit, static_argnums=(0,))
    def logpdf(self, x: ArrayLike) -> ArrayLike:
        return jax_truncnorm.logpdf(x, self._alpha, self._beta, loc=self._mu, scale=self._sigma)

    @partial(jit, static_argnums=(0,))
    def pdf(self, x: ArrayLike) -> ArrayLike:
        return jax_truncnorm.pdf(x, self._alpha, self._beta, loc=self._mu, scale=self._sigma)

    @partial(jit, static_argnums=(0,))
    def logcdf(self, x: ArrayLike) -> ArrayLike:
        return jax_truncnorm.logcdf(x, self._alpha, self._beta, loc=self._mu, scale=self._sigma)

    @partial(jit, static_argnums=(0,))
    def cdf(self, x: ArrayLike) -> ArrayLike:
        return jax_truncnorm.cdf(x, self._alpha, self._beta, loc=self._mu, scale=self._sigma)

    def rvs(self, N: int) -> Array:
        return jax.random.truncated_normal(
            self.get_key(),
            self._alpha,
            self._beta,
            shape=(N,),
            dtype=jnp.float32,
        ) * self._sigma + self._mu

    def __repr__(self) -> str:
        string = f"TruncNorm(mu={self._mu}, sigma={self._sigma}, low={self._low}, high={self._high}"
        if self._name is not None:
            string += f", name={self._name}"
        return string + ")"
