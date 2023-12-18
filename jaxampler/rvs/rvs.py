import random
from functools import partial

import jax
import jax.random
from jax import Array, jit
from jax import numpy as jnp
from jax.typing import ArrayLike


class GenericRV(object):
    """Generic random variable class."""

    def __init__(self, name: str = None) -> None:
        """Initialize the random variable.

        Parameters
        ----------
        name : str, optional
            Name of the random variable, by default None
        """
        self._name = None if name is None else name

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

    def rvs(self, N: int = 1, key: Array = None) -> Array:
        """Random variates from the distribution of size N.

        Parameters
        ----------
        N : int, optional
            Number of random variates, by default 1
        key : Array, optional
            JAX random key, by default None

        Returns
        -------
        Array
            Random variates from the distribution of size N.

        Raises
        ------
        NotImplementedError
            If the child class has not implemented this method.
        """
        raise NotImplementedError

    @staticmethod
    def get_key(key: Array = None) -> Array:
        """Get a new JAX random key.

        This method is used to generate a new JAX random key if
        the user does not provide one. The key is generated using
        the JAX random.PRNGKey function. The key is split into
        two keys, the first of which is returned. The second key
        is discarded.

        Parameters
        ----------
        key : Array, optional
            JAX random key, by default None

        Returns
        -------
        Array
            New JAX random key.
        """
        if key is None:
            new_key = jax.random.PRNGKey(random.randint(0, 1e8))
        else:
            new_key, _ = jax.random.split(key)
        return new_key

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
