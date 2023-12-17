from functools import partial

from jax import jit
from jax import numpy as jnp
from jax.typing import ArrayLike

from ..rvs import GenericRV


class DiscreteRV(GenericRV):

    def __init__(self, name: str = None) -> None:
        super().__init__(name)

    @partial(jit, static_argnums=(0,))
    def logpmf(self, *k: ArrayLike) -> ArrayLike:
        raise NotImplementedError

    @partial(jit, static_argnums=(0,))
    def pmf(self, *k: ArrayLike) -> ArrayLike:
        return jnp.exp(self.logpmf(*k))

    def __str__(self) -> str:
        return super().__str__()
