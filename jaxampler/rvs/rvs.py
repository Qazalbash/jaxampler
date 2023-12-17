import random
from functools import partial

import jax
import jax.random
from jax import Array, jit
from jax import numpy as jnp
from jax.typing import ArrayLike


class GenericRV(object):

    def __init__(self, name: str = None) -> None:
        self._name = None if name is None else name

    def check_params(self) -> None:
        raise NotImplementedError

    @partial(jit, static_argnums=(0,))
    def logcdf(self, *x: ArrayLike) -> ArrayLike:
        raise NotImplementedError

    @partial(jit, static_argnums=(0,))
    def cdf(self, *x: ArrayLike) -> ArrayLike:
        return jnp.exp(self.logcdf(*x))

    @partial(jit, static_argnums=(0,))
    def logppf(self, *x: ArrayLike) -> ArrayLike:
        raise NotImplementedError

    @partial(jit, static_argnums=(0,))
    def ppf(self, *x: ArrayLike) -> ArrayLike:
        return jnp.exp(self.logppf(*x))

    def rvs(self, N: int = 1, key: Array = None) -> Array:
        raise NotImplementedError

    @staticmethod
    def get_key(key: Array = None) -> Array:
        if key is None:
            new_key = jax.random.PRNGKey(random.randint(0, 1e6))
        else:
            new_key, _ = jax.random.split(key)
        return new_key

    def __str__(self) -> str:
        return self.__repr__()
