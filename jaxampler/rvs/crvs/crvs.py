from functools import partial

from jax import jit
from jax import numpy as jnp
from jax.typing import ArrayLike

from ..rvs import GenericRV


class ContinuousRV(GenericRV):

    def __init__(self, name: str = None) -> None:
        super().__init__(name)

    @partial(jit, static_argnums=(0,))
    def Z(self) -> ArrayLike:
        return jnp.exp(self._logZ)

    def logpdf(self, *x: ArrayLike) -> ArrayLike:
        raise NotImplementedError

    @partial(jit, static_argnums=(0,))
    def pdf(self, *x: ArrayLike) -> ArrayLike:
        return jnp.exp(self.logpdf(*x))

    def __str__(self) -> str:
        return super().__str__()
