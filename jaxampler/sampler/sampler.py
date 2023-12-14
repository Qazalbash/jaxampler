from abc import abstractmethod
from time import time

import jax
from jax import Array, jit

from ..rvs import ContinuousRV, GenericRV


class Sampler(object):

    def __init__(self) -> None:
        pass

    def check_rv(self, rv: GenericRV) -> None:
        assert isinstance(rv, GenericRV), f"rv must be a GenericRV object, got {rv}"
        assert isinstance(rv, ContinuousRV), f"rv must be a ContinuousRV object, got {rv}"
        assert hasattr(rv, "logppf") or hasattr(rv, "ppf"), f"rv must have a method called logppf or ppf"

    @abstractmethod
    def sample(self, rv: GenericRV, N: int = 1, key: Array = None) -> Array:
        raise NotImplementedError

    @staticmethod
    @jit
    def get_key() -> Array:
        return jax.random.PRNGKey(int(time()))
