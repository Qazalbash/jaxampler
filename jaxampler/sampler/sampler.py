from abc import abstractmethod
from functools import partial
from time import time

import jax
from jax import Array, jit
from jax import numpy as jnp
from jax.typing import ArrayLike

from ..rvs import ContinuousRV, GenericRV


class Sampler(object):

    def __init__(self) -> None:
        pass

    def check_rv(self, rv: GenericRV) -> None:
        assert isinstance(rv, GenericRV), f"rv must be a GenericRV object, got {rv}"
        assert isinstance(rv, ContinuousRV), f"rv must be a ContinuousRV object, got {rv}"
        assert hasattr(rv, "logppf") or hasattr(rv, "ppf"), f"rv must have a method called logppf or ppf"

    @abstractmethod
    def sample(self, rv: GenericRV, N: int = 1) -> Array:
        pass

    @staticmethod
    @jit
    def get_key() -> int:
        return jax.random.PRNGKey(int(time()))
