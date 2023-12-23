import random
from abc import abstractmethod

import jax
from jax import Array

from ..rvs import ContinuousRV, GenericRV


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
        """Get a new JAX random key.

        This method is used to generate a new JAX random key if
        the user does not provide one. The key is generated using
        the JAX random.PRNGKey function. The key is split into
        two keys, the first of which is returned. The second key
        is discarded.

        Parameters
        ----------
        key : Array, optional
            JAX random key, by default None

        Returns
        -------
        Array
            New JAX random key.
        """
        if key is None:
            new_key = jax.random.PRNGKey(random.randint(0, 1e8))
        else:
            new_key, _ = jax.random.split(key)
        return new_key
