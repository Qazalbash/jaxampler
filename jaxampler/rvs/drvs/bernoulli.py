from functools import partial

import jax
from jax import Array, jit, lax
from jax import numpy as jnp
from jax._src.lax.lax import _const as _lax_const
from jax.scipy.stats import bernoulli as jax_bernoulli
from jax.typing import ArrayLike

from .binomial import Binomial


class Bernoulli(Binomial):

    def __init__(self, p: ArrayLike, name: str = None) -> None:
        super().__init__(p, 1, name)

    def __repr__(self) -> str:
        string = f"Bernoulli(p={self._p}"
        if self._name is not None:
            string += f", name={self._name}"
        return string + ")"
