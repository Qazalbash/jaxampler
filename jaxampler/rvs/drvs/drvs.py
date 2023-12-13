from functools import partial

import jax
from jax import Array, jit
from jax import numpy as jnp
from jax.scipy.stats import beta as jax_beta
from jax.typing import ArrayLike

from ..rvs import GenericRV


class DiscreteRV(GenericRV):

    def __init__(self, name: str = None) -> None:
        super().__init__(name)

    @partial(jit, static_argnums=(0,))
    def logpmf(self, x: ArrayLike) -> ArrayLike:
        raise NotImplementedError

    @partial(jit, static_argnums=(0,))
    def pmf(self, x: ArrayLike) -> ArrayLike:
        return jnp.exp(self.logpmf(x))

    def __str__(self) -> str:
        return super().__str__()
