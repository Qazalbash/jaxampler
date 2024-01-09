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

from jax import Array
from jax import numpy as jnp
from jax.typing import ArrayLike

from .._rvs import Uniform
from ..jobj import JObj
from ..utils import jx_cast
from .montecarlogeneric import MonteCarloGenericIntegration

MCGenInt = MonteCarloGenericIntegration(name="forMonteCarloBoxIntegration")


class MonteCarloBoxIntegration(JObj):
    """Monte Carlo Integration with a uniform probability distribution.

    .. math::

        \\int_a^b h(x) dx = (b-a)\\int_a^b h(x)\\mathcal{U}(a,b) dx \\approx \\frac{b-a}{N} \\sum_{i=1}^N h(x_i)

    where :math:`x_i \\sim p(x)`. This is a special case of Monte Carlo
    integration, and is not optimized for any particular probability
    distribution.
    """

    def __init__(self, name: str = None) -> None:
        super().__init__(name)

    def compute_integral(
        self,
        h: Callable,
        low: ArrayLike,
        high: ArrayLike,
        N: int,
        key: Array = None,
    ) -> float:
        """Computes the integral of a function using Monte Carlo integration.

        Parameters
        ----------
        h : Callable
            First part of the integrand.
        low : ArrayLike
            lower bound of the integral.
        high : ArrayLike
            upper bound of the integral.
        N : int
            number of samples.
        key : Array, optional
            JAX random key, by default None

        Returns
        -------
        float
            integral of the function.
        """
        low, high = jx_cast(low, high)
        volume = jnp.prod(high - low)
        return volume * MCGenInt.compute_integral(
            h=h,
            p=Uniform(low=low, high=high),
            low=low,
            high=high,
            N=N,
            key=key,
        )

    def __repr__(self) -> str:
        string = f"MonteCarloBoxIntegration(p={self._p}, q={self._q}"
        if self._name is not None:
            string += f", name={self._name}"
        string += ")"
        return string
