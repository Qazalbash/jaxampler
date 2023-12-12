from functools import partial
from time import time

from jax import Array, jit
import jax
from jax import numpy as jnp
from jax.typing import ArrayLike


class GenericRV(object):

    def __init__(self, name: str = None) -> None:
        self._name = None if name is None else name

    @partial(jit, static_argnums=(0,))
    def logcdf(self, x: ArrayLike) -> ArrayLike:
        raise NotImplementedError

    @partial(jit, static_argnums=(0,))
    def cdf(self, x: ArrayLike) -> ArrayLike:
        return jnp.exp(self.logcdf(x))

    @partial(jit, static_argnums=(0,))
    def cdfinv(self, x: ArrayLike) -> ArrayLike:
        return jnp.exp(self.logcdfinv(x))

    def logcdfinv(self, x: ArrayLike) -> ArrayLike:
        raise NotImplementedError

    def __str__(self) -> str:
        return self.__repr__()
