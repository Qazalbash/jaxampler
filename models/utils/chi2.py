from functools import partial

import jax
from jax import Array
from jax.scipy.stats import chi2 as jax_chi2
from jax.typing import ArrayLike

from .continuousrv import ContinuousRV


class Chi2(ContinuousRV):

    def __init__(self, nu: ArrayLike, name: str = None) -> None:
        self._nu = nu
        self.check_params()
        super().__init__(name)

    def check_params(self) -> None:
        assert self._nu % 1 == 0, "nu must be an integer"

    @partial(jax.jit, static_argnums=(0,))
    def logpdf(self, x: ArrayLike) -> ArrayLike:
        return jax_chi2.logpdf(x, self._nu)

    @partial(jax.jit, static_argnums=(0,))
    def pdf(self, x: ArrayLike) -> ArrayLike:
        return jax_chi2.pdf(x, self._nu)

    @partial(jax.jit, static_argnums=(0,))
    def logcdf(self, x: ArrayLike) -> ArrayLike:
        return jax_chi2.logcdf(x, self._nu)

    @partial(jax.jit, static_argnums=(0,))
    def cdf(self, x: ArrayLike) -> ArrayLike:
        return jax_chi2.cdf(x, self._nu)

    @partial(jax.jit, static_argnums=(0,))
    def logcdfinv(self, x: ArrayLike) -> ArrayLike:
        raise NotImplementedError("Not able to find sufficient information to implement")

    def rvs(self, N: int) -> Array:
        return jax.random.chisquare(self.get_key(), self._nu, shape=(N,))

    def __repr__(self) -> str:
        string = f"Chi2(nu={self._nu}"
        if self._name is not None:
            string += f", name={self._name}"
        return string + ")"
