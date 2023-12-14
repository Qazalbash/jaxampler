from functools import partial

import jax
from jax import Array, jit
from jax import numpy as jnp
from jax.random import KeyArray
from jax.scipy.special import log_ndtr, ndtr, ndtri
from jax.typing import ArrayLike

from .crvs import ContinuousRV


class LogNormal(ContinuousRV):

    def __init__(self, mu: float = 0.0, sigma: float = 1.0, name: str = None) -> None:
        self._mu = mu
        self._sigma = sigma
        self.check_params()
        super().__init__(name)

    def check_params(self) -> None:
        assert self._sigma > 0.0, "All sigma must be greater than 0.0"

    @partial(jit, static_argnums=(0,))
    def logpdf(self, x: ArrayLike) -> ArrayLike:
        constants = -(jnp.log(self._sigma) + 0.5 * jnp.log(2 * jnp.pi))
        logpdf_val = jnp.where(
            x <= 0,
            -jnp.inf,
            constants - jnp.log(x) - (0.5 * jnp.power(self._sigma, -2)) * jnp.power((jnp.log(x) - self._mu), 2),
        )
        return logpdf_val

    @partial(jit, static_argnums=(0,))
    def logcdf(self, x: ArrayLike) -> ArrayLike:
        return log_ndtr((jnp.log(x) - self._mu) / self._sigma)

    @partial(jit, static_argnums=(0,))
    def cdf(self, x: ArrayLike) -> ArrayLike:
        return ndtr((jnp.log(x) - self._mu) / self._sigma)

    @partial(jit, static_argnums=(0,))
    def ppf(self, x: ArrayLike) -> ArrayLike:
        return jnp.exp(self._mu + self._sigma * ndtri(x))

    def rvs(self, N: int = 1, key: KeyArray = None) -> Array:
        # return jax.random.lognormal(self.get_key(), shape=(N,)) * self._sigma + self._mu
        if key is None:
            key = self.get_key()
        U = jax.random.uniform(key, shape=(N,))
        return self.ppf(U)

    def __repr__(self) -> str:
        string = f"LogNormal(mu={self._mu}, sigma={self._sigma}"
        if self._name is not None:
            string += f", name={self._name}"
        return string + ")"
