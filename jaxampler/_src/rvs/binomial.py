#  Copyright 2023 The Jaxampler Authors
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from __future__ import annotations

from functools import partial
from typing import Any, Optional

import jax
from jax import Array, jit, numpy as jnp
from jax.scipy.special import betainc
from jax.scipy.stats import binom as jax_binom

from ..typing import Numeric
from ..utils import jx_cast
from .drvs import DiscreteRV


class Binomial(DiscreteRV):
    r"""Binomial random variable
    .. math::
        X\sim Bin(p,n) \iff P(X=x|p,n)=\binom{n}{x}p^{x}(1-p)^{n-x}
    """

    def __init__(self, p: Numeric | Any, n: Numeric | Any, name: Optional[str] = None) -> None:
        """
        :param p: Probability of success
        :param n: Number of trials
        :param name: Name of the random variable
        """
        shape, self._p, self._n = jx_cast(p, n)
        self.check_params()
        self._q = 1.0 - self._p
        super().__init__(name=name, shape=shape)

    def check_params(self) -> None:
        """Check the parameters of the random variable."""
        assert jnp.all(self._p >= 0.0) and jnp.all(self._p <= 1.0), "p must be in [0, 1]"
        assert jnp.all(self._n.dtype == jnp.int32), "n must be an integer"
        assert jnp.all(self._n > 0), "n must be positive"

    @partial(jit, static_argnums=(0,))
    def logpmf_x(self, x: Numeric) -> Numeric:
        return jax_binom.logpmf(x, self._n, self._p)

    @partial(jit, static_argnums=(0,))
    def pmf_x(self, x: Numeric) -> Numeric:
        return jax_binom.pmf(x, self._n, self._p)

    @partial(jit, static_argnums=(0,))
    def logcdf_x(self, x: Numeric) -> Numeric:
        return jnp.log(self.cdf_x(x))

    @partial(jit, static_argnums=(0,))
    def cdf_x(self, x: Numeric) -> Numeric:
        floor_x = jnp.floor(x)
        cond = [x < 0, x >= self._n, jnp.logical_and(x >= 0, x < self._n)]
        return jnp.select(cond, [0.0, 1.0, betainc(self._n - floor_x, floor_x + 1, self._q)])

    def rvs(self, shape: tuple[int, ...], key: Optional[Array] = None) -> Array:
        if key is None:
            key = self.get_key()
        new_shape = shape + self._shape
        return jax.random.binomial(key=key, n=self._n, p=self._p, shape=new_shape)

    def __repr__(self) -> str:
        string = f"Binomial(p={self._p}, n={self._n}"
        if self._name is not None:
            string += f", name={self._name}"
        string += ")"
        return string
