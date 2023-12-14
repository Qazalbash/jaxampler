from functools import partial

import jax
from jax import Array, jit
from jax import numpy as jnp
from jax.typing import ArrayLike

from ..rvs import GenericRV
from .sampler import Sampler


class InverseTransformSampler(Sampler):

    def __init__(self) -> None:
        super().__init__()

    def sample(self, rv: GenericRV, N: int = 1) -> Array:
        self.check_rv(rv)
        U = jax.random.uniform(self.get_key(), shape=(N,))
        return rv.ppf(U)
