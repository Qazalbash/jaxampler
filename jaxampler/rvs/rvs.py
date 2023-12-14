from functools import partial
from time import time

import jax
from jax import Array, jit
from jax import numpy as jnp
from jax.random import KeyArray
from jax.typing import ArrayLike


class GenericRV(object):

    def __init__(self, name: str = None) -> None:
        self._name = None if name is None else name

    def check_params(self) -> None:
        raise NotImplementedError

    @partial(jit, static_argnums=(0,))
    def logcdf(self, x: ArrayLike) -> ArrayLike:
        raise NotImplementedError

    @partial(jit, static_argnums=(0,))
    def cdf(self, x: ArrayLike) -> ArrayLike:
        return jnp.exp(self.logcdf(x))

    @partial(jit, static_argnums=(0,))
    def logppf(self, x: ArrayLike) -> ArrayLike:
        raise NotImplementedError

    @partial(jit, static_argnums=(0,))
    def ppf(self, x: ArrayLike) -> ArrayLike:
        return jnp.exp(self.logppf(x))

    def rvs(self, N: int = 1, key: KeyArray = None) -> Array:
        raise NotImplementedError

    @staticmethod
    @jit
    def get_key() -> KeyArray:
        return jax.random.PRNGKey(int(time()))

    def __str__(self) -> str:
        return self.__repr__()
