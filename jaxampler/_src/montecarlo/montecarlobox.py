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

from typing import Callable, Optional

from jax import Array, numpy as jnp

from jaxampler._src.montecarlo.integration import Integration
from jaxampler._src.montecarlo.montecarlogeneric import MonteCarloGenericIntegration
from jaxampler._src.rvs import Uniform


MCGenInt = MonteCarloGenericIntegration(name="forMonteCarloBoxIntegration")


class MonteCarloBoxIntegration(Integration):
    """Monte Carlo Integration with a uniform probability distribution.

    .. math::

        \\int_a^b h(x) dx = (b-a)\\int_a^b h(x)\\mathcal{U}(a,b) dx \\approx \\frac{b-a}{N} \\sum_{i=1}^N h(x_i)

    where :math:`x_i \\sim p(x)`. This is a special case of Monte Carlo
    integration, and is not optimized for any particular probability
    distribution.
    """

    def __init__(self, name: Optional[str] = None) -> None:
        super().__init__(name)

    def compute_integral(
        self,
        h: Callable,
        low: Array,
        high: Array,
        N: int,
        *args,
        key: Optional[Array] = None,
        **kwargs,
    ) -> Array:
        """Computes the integral of a function using Monte Carlo integration.

        Parameters
        ----------
        h : Callable
            First part of the integrand.
        low : Numeric
            lower bound of the integral.
        high : Numeric
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
        integral = MCGenInt.compute_integral(
            h=h,
            p=Uniform(low=low, high=high),
            low=low,
            high=high,
            N=N,
            key=key,
        )
        volume = jnp.prod(high - low, axis=0, dtype=jnp.float32)
        return volume * integral

    def __repr__(self) -> str:
        string = "MonteCarloBoxIntegration("
        if self._name is not None:
            string += f", name={self._name}"
        string += ")"
        return string
