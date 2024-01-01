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

from jax import Array, jit
from jax import numpy as jnp
from jax import vmap
from jax.typing import ArrayLike

from ..utils import new_prn_key


class GenericRV(object):
    """Generic random variable class."""

    def __init__(self, name: str = None) -> None:
        """Initialize the random variable.

        Parameters
        ----------
        name : str, optional
            Name of the random variable, by default None
        """
        self._name = name

    def check_params(self) -> None:
        """Check the parameters of the random variable.

        This method should be implemented by the child class.

        Raises
        ------
        NotImplementedError
            If the child class has not implemented this method.
        """
        raise NotImplementedError

    @partial(jit, static_argnums=(0,))
    def logcdf(self, *x: ArrayLike) -> ArrayLike:
        """Logarithm of the cumulative distribution function.
        
        Parameters
        ----------
        *x : ArrayLike
            Input values.

        Returns
        -------
        ArrayLike
            The logarithm of the cumulative distribution function.

        Raises
        ------
        NotImplementedError
            If the child class has not implemented this method.
        """
        raise NotImplementedError

    @partial(jit, static_argnums=(0,))
    def cdf(self, *x: ArrayLike) -> ArrayLike:
        """Cumulative distribution function.
        
        Parameters
        ----------
        *x : ArrayLike
            Input values.

        Returns
        -------
        ArrayLike
            The cumulative distribution function evaluated at x.
        """
        return jnp.exp(self.logcdf(*x))

    @partial(jit, static_argnums=(0,))
    def logppf(self, *x: ArrayLike) -> ArrayLike:
        """Logarithm of the percent point function.

        Returns
        -------
        ArrayLike
            The logarithm of the percent point function evaluated at x.

        Raises
        ------
        NotImplementedError
            If the child class has not implemented this method.
        """
        raise NotImplementedError

    @partial(jit, static_argnums=(0,))
    def ppf(self, *x: ArrayLike) -> ArrayLike:
        """Percent point function.

        Returns
        -------
        ArrayLike
            The percent point function evaluated at x.
        """
        return jnp.exp(self.logppf(*x))

    def rvs(self, shape: tuple[int, ...], key: Array = None) -> Array:
        """Random variates from the distribution of size N.

        Parameters
        ----------
        shape : int, optional
            shape of the rvs output
        key : Array, optional
            JAX random key, by default None

        Returns
        -------
        Array
            Random variates from the distribution of shape `shape`.

        Raises
        ------
        NotImplementedError
            If the child class has not implemented this method.
        """
        raise NotImplementedError

    @staticmethod
    def get_key(key: Array = None) -> Array:
        return new_prn_key(key)

    def __str__(self) -> str:
        """String representation of the random variable.

        Returns
        -------
        str
            String representation of the random variable.
        """
        return self.__repr__()

    def __repr__(self) -> str:
        """String representation of the random variable.

        Returns
        -------
        str
            String representation of the random variable.

        Raises
        ------
        NotImplementedError
            If the child class has not implemented this method.
        """
        raise NotImplementedError
