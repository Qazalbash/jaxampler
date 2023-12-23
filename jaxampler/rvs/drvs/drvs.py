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
from jax.typing import ArrayLike

from ..rvs import GenericRV


class DiscreteRV(GenericRV):

    def __init__(self, name: str = None) -> None:
        super().__init__(name)

    @partial(jit, static_argnums=(0,))
    def logpmf(self, *k: ArrayLike) -> ArrayLike:
        """Logarithm of the probability mass function.

        Parameters
        ----------
        *k : ArrayLike
            Input values.   

        Returns
        -------
        ArrayLike
            Logarithm of the probability mass function evaluated at *k.

        Raises
        ------
        NotImplementedError
            If the child class has not implemented this method.
        """
        raise NotImplementedError

    @partial(jit, static_argnums=(0,))
    def pmf(self, *k: ArrayLike) -> ArrayLike:
        """Probability mass function.

        Parameters
        ----------
        *k : ArrayLike
            Input values.

        Returns
        -------
        ArrayLike
            Probability mass function evaluated at *k.
        """
        return jnp.exp(self.logpmf(*k))

    def __str__(self) -> str:
        return super().__str__()
