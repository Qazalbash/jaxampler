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

import jax
from jax import Array, jit
from jax import numpy as jnp
from jax.scipy.stats import binom as jax_binom
from jax.typing import ArrayLike

from ...utils import jx_cast
from .drvs import DiscreteRV


class Binomial(DiscreteRV):
    """Binomial random variable"""

    def __init__(self, p: ArrayLike, n: int, name: str = None) -> None:
        """Initialize the Binomial random variable.

        Parameters
        ----------
        p : ArrayLike
            Probability of success.
        n : int
            Number of trials.
        name : str, optional
            Name of the random variable, by default None
        """
        self._p, self._n = jx_cast(p, n)
        self.check_params()
        self._q = 1.0 - p
        super().__init__(name)

    def check_params(self) -> None:
        """Check the parameters of the random variable."""
        assert jnp.all(self._p >= 0.0) and jnp.all(self._p <= 1.0), "p must be in [0, 1]"
        assert jnp.all(self._n.dtype == jnp.int32), "n must be an integer"
        assert jnp.all(self._n > 0), "n must be positive"

    @partial(jit, static_argnums=(0))
    def logpmf_x(self, x: ArrayLike) -> ArrayLike:
        return jax_binom.logpmf(x, self._n, self._p)

    @partial(jit, static_argnums=(0,))
    def pmf_x(self, x: ArrayLike) -> ArrayLike:
        return jax_binom.pmf(x, self._n, self._p)

    @partial(jit, static_argnums=(0,))
    def logcdf_x(self, x: ArrayLike) -> ArrayLike:
        return jnp.log(self.cdf_x(x))

    @partial(jit, static_argnums=(0,))
    def cdf_x(self, x: ArrayLike) -> ArrayLike:
        xx = jnp.arange(0, self._n + 1, dtype=jnp.int32)
        complete_cdf = jnp.cumsum(self.pmf_x(xx))
        cond = [x < 0, x >= self._n, jnp.logical_and(x >= 0, x < self._n)]
        return jnp.select(cond, [0.0, 1.0, complete_cdf[x]])

    def rvs(self, shape: tuple[int, ...], key: Array = None) -> Array:
        if key is None:
            key = self.get_key(key)
        return jax.random.binomial(key=key, n=self._n, p=self._p, shape=shape)

    def __repr__(self) -> str:
        string = f"Binomial(p={self._p}, n={self._n}"
        if self._name is not None:
            string += f", name={self._name}"
        string += ")"
        return string
