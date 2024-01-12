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

import jax
from jax import Array, jit, numpy as jnp
from jax.scipy.special import betainc
from jax.scipy.stats import t as jax_t

from jaxampler._src.rvs.crvs import ContinuousRV
from jaxampler._src.typing import Numeric
from jaxampler._src.utils import jx_cast


class StudentT(ContinuousRV):
    def __init__(self, nu: Numeric, name: Optional[str] = None) -> None:
        shape, self._nu = jx_cast(nu)
        self.check_params()
        super().__init__(name=name, shape=shape)

    def check_params(self) -> None:
        assert jnp.all(self._nu > 0.0), "nu must be positive"

    @partial(jit, static_argnums=(0,))
    def logpdf_x(self, x: Numeric) -> Numeric:
        return jax_t.logpdf(x, self._nu)

    @partial(jit, static_argnums=(0,))
    def pdf_x(self, x: Numeric) -> Numeric:
        return jax_t.pdf(x, self._nu)

    @partial(jit, static_argnums=(0,))
    def cdf_x(self, x: Numeric) -> Numeric:
        return 1 - 0.5 * betainc(self._nu * 0.5, 0.5, 1 / (1 + (jnp.power(x, 2) / self._nu)))

    @partial(jit, static_argnums=(0,))
    def ppf_x(self, x: Numeric) -> Numeric:
        """A method is addressed in this paper https://www.homepages.ucl.ac.uk/~ucahwts/lgsnotes/JCF_Student.pdf"""
        raise NotImplementedError

    def rvs(self, shape: tuple[int, ...], key: Optional[Array] = None) -> Array:
        if key is None:
            key = self.get_key()
        new_shape = shape + self._shape
        return jax.random.t(key=key, df=self._nu, shape=new_shape)

    def __repr__(self) -> str:
        string = f"StudentT(nu={self._nu}"
        if self._name is not None:
            string += f", name={self._name}"
        string += ")"
        return string
