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
from jax import vmap
from jax.typing import ArrayLike

from .._rvs import ContinuousRV
from ..utils import jx_cast
from .integration import Integration


class MonteCarloGenericIntegration(Integration):
    """Monte Carlo Integration with a generic probability distribution.

    .. math::

        \\int_a^b h(x) p(x) dx \\approx \\frac{1}{N} \\sum_{i=1}^N h(x_i)

    where :math:`x_i \\sim p(x)`. This is a generic implementation of Monte Carlo
    integration, and is not optimized for any particular probability distribution.
    """

    def __init__(self, name: str = None) -> None:
        super().__init__(name)

    def compute_integral(
        self,
        h: Callable,
        p: ContinuousRV,
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
        p : ContinuousRV
            Probability distribution. It is part of the integrand.
        low : ArrayLike
            lower bound of the integral.
        high : ArrayLike
            upper bound of the integral.
        N : int
            Number of samples.
        key : Array, optional
            JAX random key, by default None

        Returns
        -------
        float
            Integral of the function.
        """
        if key is None:
            key = self.get_key()
        low, high = jx_cast(low, high)
        p_rv = p.rvs(shape=(N,) + low.shape, key=key)
        low, high = jx_cast(low, high)
        p_rv = p_rv[(p_rv >= low) & (p_rv <= high)]
        hx = vmap(h)(p_rv)
        return jnp.mean(hx)

    def __repr__(self) -> str:
        string = "MonteCarloGenericIntegration("
        if self._name is not None:
            string += f"name={self._name}"
        string += ")"
        return string
