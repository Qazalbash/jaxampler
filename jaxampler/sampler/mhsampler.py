# Copyright 2023 The Jaxampler Authors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Callable

from jax import Array
from jax import numpy as jnp
from jax.random import uniform
from jax.typing import ArrayLike
from tqdm import tqdm, trange

from ..rvs import ContinuousRV
from .sampler import Sampler


class MetropolisHastingSampler(Sampler):
    """Metropolis-Hasting Sampler Class"""

    def __init__(self, name: str = None) -> None:
        super().__init__(name)

    def _walk(self,
              q: Callable[[Any], ContinuousRV],
              alpha: Callable,
              xt: ArrayLike,
              N: int,
              key: Array = None) -> Array:
        """single chain walk 

        This function is used to generate a single chain of samples from the
        target distribution.

        Parameters
        ----------
        q : Callable[[Any], ContinuousRV]
            Proxy distribution
        alpha : Callable
            Acceptance probability function
        xt : ArrayLike
            Initial value
        N : int
            Number of samples
        key : Array, optional
            JAX PRNGKey, by default None

        Returns
        -------
        Array
            Samples from the target distribution
        """
        if key is None:
            key = self.get_key(key)
        samples = jnp.empty((N,))
        t = 0
        while t < N:
            x_prop = q(xt).rvs(shape=(), key=key)
            key = self.get_key(key)
            u = uniform(key)
            if u < alpha(xt, x_prop):
                xt = x_prop
                samples = samples.at[t].set(xt)
                t += 1
            key = self.get_key(key)
        return samples

    def sample(self,
               p: ContinuousRV,
               q: Callable[[Any], ContinuousRV],
               burn_in: int,
               n_chains: int,
               x_curr: Array,
               N: int,
               key: Array = None,
               hasting_ratio: bool = False) -> Array:
        """Sample function for Metropolis-Hasting Sampler

        First, the sampler will run a burn-in phase to get the chain to
        stationarity. Then, the sampler will run the sampling phase to generate
        samples from the target distribution.

        Parameters
        ----------
        p : ContinuousRV
            Target distribution
        q : Callable[[Any], ContinuousRV]
            Proxy distribution
        burn_in : int
            Burn-in phase
        n_chains : int
            Number of chains
        x_curr : Array
            Initial values
        N : int
            Number of samples
        key : Array, optional
            JAX PRNGKey, by default None
        hasting_ratio : bool, optional
            Whether to use the Hasting Ratio or not, by default False

        Returns
        -------
        Array
            Samples from the target distribution
        """
        x_curr = jnp.asarray(x_curr)
        assert x_curr.shape == (n_chains,)

        if hasting_ratio:
            alpha = lambda x1, x2: ((p.pdf_x(x2) / p.pdf_x(x1)) * (q(x1).pdf_x(x2) / q(x2).pdf_x(x1))).clip(0.0, 1.0)
        else:
            alpha = lambda x1, x2: (p.pdf_x(x2) / p.pdf_x(x1)).clip(0.0, 1.0)

        if key is None:
            key = self.get_key(key)

        for _ in trange(
                burn_in,
                desc="Burn-in",
                unit="samples",
        ):
            x_curr = q(x_curr).rvs(shape=(n_chains,), key=key)
            key = self.get_key(key)

        pbar = tqdm(
            total=N * n_chains,
            desc="Sampling",
            unit="samples",
        )

        T = jnp.zeros((n_chains,), dtype=jnp.int32)
        samples = jnp.empty((N, n_chains))

        while jnp.all(T < N):
            x_prop = q(x_curr).rvs(shape=(), key=key)
            key = self.get_key(key)
            u = uniform(key, (n_chains,))
            cond = u < alpha(x_curr, x_prop)
            x_curr = jnp.where(cond == True, x_prop, x_curr)
            for i in range(n_chains):
                if cond[i]:
                    samples = samples.at[T[i], i].set(x_prop[i])
            T += cond
            key = self.get_key(key)
            t = int(cond.sum())
            pbar.update(t)

        pbar.display("Parallel Sampling Finished")

        for i in range(n_chains):
            if T[i] < N:
                samples = samples.at[T[i]:, i].set(self._walk(
                    q,
                    alpha,
                    x_curr[i],
                    N - T[i],
                    key,
                ))
                key = self.get_key(key)
                t = int(N - T[i])
                pbar.update(t)

        pbar.close()

        return samples
