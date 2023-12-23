import jax
from jax import Array
from jax import numpy as jnp
from matplotlib import pyplot as plt

from ..rvs import ContinuousRV
from .sampler import Sampler


class AdaptiveAcceptRejectSampler(Sampler):

    def __init__(self) -> None:
        super().__init__()

    def sample(self,
               target_rv: ContinuousRV,
               proposal_rv: ContinuousRV,
               scale: int = 1,
               N: int = 1,
               key: Array = None,
               scatter_plot: bool = False) -> Array:
        self.check_rv(target_rv)
        self.check_rv(proposal_rv)
        samples = jnp.array([])
        N_res = N
        while N_res != 0:
            N_res = N - len(samples)

            key = self.get_key(key)
            V = proposal_rv.rvs(N_res, key)
            pdf_ratio = target_rv.pdf(V)

            key = self.get_key(key)
            U_scaled = jax.random.uniform(
                key,
                shape=(N_res,),
                minval=0.0,
                maxval=scale * proposal_rv.pdf(V),
            )

            accept = U_scaled <= pdf_ratio
            samples = jnp.concatenate([samples, V[accept]])

        if scatter_plot:
            plt.scatter(
                V,
                U_scaled,
                c=accept,
                cmap="viridis",
                alpha=0.5,
                label="Accept/Reject samples",
            )
        return samples
