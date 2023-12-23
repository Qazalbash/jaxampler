from abc import abstractmethod

from jax import Array

from ..rvs import ContinuousRV, GenericRV
from ..utils import new_prn_key


class Sampler(object):

    def __init__(self) -> None:
        pass

    def check_rv(self, rv: GenericRV) -> None:
        assert isinstance(rv, GenericRV), f"rv must be a GenericRV object, got {rv}"
        assert isinstance(rv, ContinuousRV), f"rv must be a ContinuousRV object, got {rv}"

    @abstractmethod
    def sample(self, rv: GenericRV, N: int = 1, key: Array = None) -> Array:
        raise NotImplementedError

    @staticmethod
    def get_key(key: Array = None) -> Array:
        return new_prn_key(key)
