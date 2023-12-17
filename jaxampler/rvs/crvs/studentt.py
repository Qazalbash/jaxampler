from functools import partial

from jax import jit
from jax import numpy as jnp
from jax.scipy.special import betainc
from jax.scipy.stats import t as jax_t
from jax.typing import ArrayLike

from .crvs import ContinuousRV


class StudentT(ContinuousRV):

    def __init__(self, nu: ArrayLike, name: str = None) -> None:
        self._nu = nu
        self.check_params()
        super().__init__(name)

    def check_params(self) -> None:
        assert jnp.all(self._nu > 0.0), "nu must be positive"

    @partial(jit, static_argnums=(0,))
    def logpdf(self, x: ArrayLike) -> ArrayLike:
        return jax_t.logpdf(x, self._nu)

    @partial(jit, static_argnums=(0,))
    def pdf(self, x: ArrayLike) -> ArrayLike:
        return jax_t.pdf(x, self._nu)

    @partial(jit, static_argnums=(0,))
    def cdf(self, x: ArrayLike) -> ArrayLike:
        return 1 - 0.5 * betainc(self._nu * 0.5, 0.5, 1 / (1 + jnp.power(x, 2) / self._nu))

    @partial(jit, static_argnums=(0,))
    def ppf(self, x: ArrayLike) -> ArrayLike:
        """A method is addressed in this paper https://www.homepages.ucl.ac.uk/~ucahwts/lgsnotes/JCF_Student.pdf"""
        raise NotImplementedError

    def __repr__(self) -> str:
        string = f"StudentT(nu={self._nu}"
        if self._name is not None:
            string += f", name={self._name}"
        return string + ")"
