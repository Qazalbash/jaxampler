from functools import partial
from time import time

import jax
from jax import Array, jit
from jax import numpy as jnp
from jax.typing import ArrayLike

from ..rvs import GenericRV


class ContinuousRV(GenericRV):

    def __init__(self, name: str = None) -> None:
        self._logZ = None
        super().__init__(name)

    def check_params(self) -> None:
        raise NotImplementedError

    def logZ(self) -> ArrayLike:
        raise NotImplementedError

    @partial(jit, static_argnums=(0,))
    def Z(self) -> ArrayLike:
        return jnp.exp(self._logZ)

    def logpdf(self, x: ArrayLike) -> ArrayLike:
        raise NotImplementedError

    @partial(jit, static_argnums=(0,))
    def pdf(self, x: ArrayLike) -> ArrayLike:
        return jnp.exp(self.logpdf(x))

    def logrvs(self, N: int) -> Array:
        raise NotImplementedError

    def rvs(self, N: int = 1) -> Array:
        return jnp.exp(self.logrvs(N))

    def __str__(self) -> str:
        return super().__str__()

    @partial(jit, static_argnums=(0,))
    def get_key(self) -> int:
        return jax.random.PRNGKey(int(time()))
