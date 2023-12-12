from functools import partial
from time import time

import jax
from jax import Array
from jax import numpy as jnp
from jax.typing import ArrayLike


class ContinuousRV(object):

    def __init__(self, name: str = None) -> None:
        self._logZ = None
        self._name = None if name is None else name

    def check_params(self) -> None:
        raise NotImplementedError

    def logZ(self) -> ArrayLike:
        raise NotImplementedError

    @partial(jax.jit, static_argnums=(0,))
    def Z(self) -> ArrayLike:
        return jnp.exp(self._logZ)

    def logpdf(self, x: ArrayLike) -> ArrayLike:
        raise NotImplementedError

    @partial(jax.jit, static_argnums=(0,))
    def pdf(self, x: ArrayLike) -> ArrayLike:
        return jnp.exp(self.logpdf(x))

    def logcdf(self, x: ArrayLike) -> ArrayLike:
        raise NotImplementedError

    @partial(jax.jit, static_argnums=(0,))
    def cdf(self, x: ArrayLike) -> ArrayLike:
        return jnp.exp(self.logcdf(x))

    @partial(jax.jit, static_argnums=(0,))
    def cdfinv(self, x: ArrayLike) -> ArrayLike:
        return jnp.exp(self.logcdfinv(x))

    def logcdfinv(self, x: ArrayLike) -> ArrayLike:
        raise NotImplementedError

    def logrvs(self, N: int) -> Array:
        raise NotImplementedError

    def rvs(self, N: int = 1) -> Array:
        return jnp.exp(self.logrvs(N))

    def __str__(self) -> str:
        return self.__repr__()

    @partial(jax.jit, static_argnums=(0,))
    def get_key(self) -> int:
        return jax.random.PRNGKey(int(time()))
