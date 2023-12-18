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
