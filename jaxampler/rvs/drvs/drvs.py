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

from functools import partial

from jax import jit
from jax import numpy as jnp
from jax import vmap
from jax.typing import ArrayLike

from ..rvs import GenericRV


class DiscreteRV(GenericRV):

    def __init__(self, name: str = None, shape: tuple[int, ...] = None) -> None:
        super().__init__(name=name, shape=shape)

    # POINT VALUED

    @partial(jit, static_argnums=(0,))
    def logpmf_x(self, *k: ArrayLike) -> ArrayLike:
        raise NotImplementedError

    @partial(jit, static_argnums=(0,))
    def pmf_x(self, *k: ArrayLike) -> ArrayLike:
        return jnp.exp(self.logpmf_x(*k))

    # VECTOR VALUED

    @partial(jit, static_argnums=(0,))
    def logpmf_v(self, *k: ArrayLike) -> ArrayLike:
        return vmap(self.logpmf_x, in_axes=0)(*k)

    @partial(jit, static_argnums=(0,))
    def pmf_v(self, *k: ArrayLike) -> ArrayLike:
        return jnp.exp(self.logpmf_v(*k))
