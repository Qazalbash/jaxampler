from functools import partial

import jax
from jax import numpy as jnp
from jax import Array, jit
from jax.scipy.stats import beta as jax_beta
from jax.typing import ArrayLike

from .drvs import DiscreteRV


class Poisson(DiscreteRV):

    def __init__(self) -> None:
        super().__init__()
