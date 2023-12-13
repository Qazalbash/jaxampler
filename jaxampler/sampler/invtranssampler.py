from functools import partial

import jax
from jax import Array, jit
from jax import numpy as jnp
from jax.typing import ArrayLike

from .sampler import Sampler


class InverseTransformSampler(Sampler):

    def __init__(self) -> None:
        super().__init__()
