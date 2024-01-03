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

from typing import Callable

import tqdm
from jax import Array
from jax import numpy as jnp
from jax.random import uniform
from jax.typing import ArrayLike

from .sampler import Sampler


class MetropolisHastingSampler(Sampler):

    def __init__(self) -> None:
        super().__init__()

    def _walk(
        self,
        q_rvs: Callable,
        alpha: Callable,
        xt: ArrayLike,
        N: int,
        key: Array = None,
    ) -> Array:
        if key is None:
            key = self.get_key(key)
        samples = jnp.empty((N,))
        t = 0
        while t < N:
            x_prop = q_rvs(xt, (), key)
            key = self.get_key(key)
            u = uniform(key)
            key = self.get_key(key)
            if u < alpha(xt, x_prop):
                xt = x_prop
                samples = samples.at[t].set(xt)
                t += 1
        return samples

    def sample(self,
               f_pdf: Callable,
               q_pdf: Callable,
               q_rvs: Callable,
               burn_in: int,
               n_chains: int,
               x_curr: Array,
               N: int,
               key: Array = None,
               hasting_ratio: bool = False) -> Array:

        x_curr = jnp.asarray(x_curr)
        assert x_curr.shape == (n_chains,)

        if hasting_ratio:
            alpha = lambda x1, x2: ((f_pdf(x2) / f_pdf(x1)) * (q_pdf(x1, x2) / q_pdf(x2, x1))).clip(0.0, 1.0)
        else:
            alpha = lambda x1, x2: (f_pdf(x2) / f_pdf(x1)).clip(0.0, 1.0)

        if key is None:
            key = self.get_key(key)

        for _ in tqdm.trange(burn_in, desc="Burn-in"):
            x_curr = q_rvs(x_curr, (n_chains,), key)
            key = self.get_key(key)

        return jnp.column_stack(
            [self._walk(
                q_rvs,
                alpha,
                x_curr.at[i].get(),
                N,
                key,
            ) for i in tqdm.trange(n_chains, desc="Sampling")])
