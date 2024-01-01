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

from functools import partial

from jax import jit
from jax import numpy as jnp
from jax import vmap
from jax.scipy.special import betainc
from jax.scipy.stats import t as jax_t
from jax.typing import ArrayLike

from ...utils import jx_cast
from .crvs import ContinuousRV


class StudentT(ContinuousRV):

    def __init__(self, nu: ArrayLike, name: str = None) -> None:
        self._nu, = jx_cast(nu)
        self.check_params()
        super().__init__(name)

    def check_params(self) -> None:
        assert jnp.all(self._nu > 0.0), "nu must be positive"

    @partial(jit, static_argnums=(0,))
    def logpdf(self, x: ArrayLike) -> ArrayLike:
        return vmap(lambda xx: jax_t.logpdf(xx, self._nu))(x)

    @partial(jit, static_argnums=(0,))
    def pdf(self, x: ArrayLike) -> ArrayLike:
        return vmap(lambda xx: jax_t.pdf(xx, self._nu))(x)

    @partial(jit, static_argnums=(0,))
    def cdf(self, x: ArrayLike) -> ArrayLike:
        return vmap(lambda xx: 1 - 0.5 * betainc(
            self._nu * 0.5,
            0.5,
            1 / (1 + (jnp.power(xx, 2) / self._nu)),
        ))(x)

    @partial(jit, static_argnums=(0,))
    def ppf(self, x: ArrayLike) -> ArrayLike:
        """A method is addressed in this paper https://www.homepages.ucl.ac.uk/~ucahwts/lgsnotes/JCF_Student.pdf"""
        raise NotImplementedError

    def __repr__(self) -> str:
        string = f"StudentT(nu={self._nu}"
        if self._name is not None:
            string += f", name={self._name}"
        string += ")"
        return string
