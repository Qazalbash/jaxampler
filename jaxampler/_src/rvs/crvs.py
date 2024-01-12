# Copyright 2023 The Jaxampler Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import partial
from typing import Optional

from jax import jit, numpy as jnp, vmap

from jaxampler._src.typing import Numeric

from .rvs import GenericRV


class ContinuousRV(GenericRV):
    def __init__(self, name: Optional[str] = None, shape: tuple[int, ...] = ()) -> None:
        if name is None:
            name = ""
        super().__init__(name=name, shape=shape)

    # POINT VALUED

    @partial(jit, static_argnums=(0,))
    def logpdf_x(self, *x: Numeric) -> Numeric:
        raise NotImplementedError

    @partial(jit, static_argnums=(0,))
    def pdf_x(self, *x: Numeric) -> Numeric:
        return jnp.exp(self.logpdf_x(*x))

    # VECTOR VALUED

    @partial(jit, static_argnums=(0,))
    def logpdf_v(self, *x: Numeric) -> Numeric:
        return vmap(self.logpdf_x, in_axes=0)(*x)

    @partial(jit, static_argnums=(0,))
    def pdf_v(self, *x: Numeric) -> Numeric:
        return jnp.exp(self.logpdf_v(*x))
