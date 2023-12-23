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
