import jax
from jax import Array
from matplotlib import pyplot as plt

from ..rvs import ContinuousRV
from .sampler import Sampler


class InverseTransformSampler(Sampler):

    def __init__(self) -> None:
        super().__init__()

    def sample(self, rv: ContinuousRV, N: int = 1, key: Array = None, scatter_plot: bool = False) -> Array:
        self.check_rv(rv)
        if key is None:
            key = self.get_key(key)
        U = jax.random.uniform(key, shape=(N,))
        samples = rv.ppf(U)
        if scatter_plot:
            plt.scatter(
                U,
                rv.cdf(U),
                c=samples,
                cmap="viridis",
                alpha=0.5,
                label="Inverse Transform samples",
            )
        return samples
