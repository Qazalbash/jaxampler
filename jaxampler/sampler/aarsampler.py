# Copyright 2023 The JAXampler Authors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
        if key is None:
            key = self.get_key(key)
        samples = jnp.array([])
        N_res = N
        while N_res != 0:
            N_res = N - len(samples)

            V = proposal_rv.rvs(N_res, key)
            key = self.get_key(key)
            pdf_ratio = target_rv.pdf(V)

            U_scaled = jax.random.uniform(
                key,
                shape=(N_res,),
                minval=0.0,
                maxval=scale * proposal_rv.pdf(V),
            )

            accept = U_scaled <= pdf_ratio
            samples = jnp.concatenate([samples, V[accept]])

            key = self.get_key(key)

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
