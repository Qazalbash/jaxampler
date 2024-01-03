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

from jax import Array
from jax import numpy as jnp
from jax.typing import ArrayLike

from ..rvs import ContinuousRV, Uniform
from ..utils import jx_cast
from .integration import Integration


class MonteCarloBoxIntegration(Integration):

    def __init__(
        self,
        p: ContinuousRV,
        name: str = None,
    ) -> None:
        self._p = p
        self._q = lambda l, h: Uniform(low=l, high=h)
        super().__init__(name)

    def check_params(self) -> None:
        assert type(self._p) == ContinuousRV

    def compute_integral(self, N: int, low: Array, high: Array, key: Array = None) -> ArrayLike:
        low, high = jx_cast(low, high)
        rvs = self._q(low, high).rvs(shape=(N,) + low.shape, key=self.get_key(key))
        mu = self._p.pdf_v(rvs)
        mu = mu.mean()
        volume = jnp.prod(high - low)
        return mu * volume

    def __repr__(self) -> str:
        string = f"MonteCarloIntegration(p={self._p}, q={self._q}"
        if self._name is not None:
            string += f", name={self._name}"
        string += ")"
        return string
