import jax
from jax import Array

from ..rvs import ContinuousRV
from .sampler import Sampler


class InverseTransformSampler(Sampler):

    def __init__(self) -> None:
        super().__init__()

    def sample(self, rv: ContinuousRV, N: int = 1, key: Array = None) -> Array:
        self.check_rv(rv)
        if key is None:
            key = self.get_key()
        U = jax.random.uniform(key, shape=(N,))
        return rv.ppf(U)
