#  Copyright 2023 The Jaxampler Authors
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from __future__ import annotations

from functools import partial
from typing import Any, Optional

import jax
from jax import Array, jit, numpy as jnp
from jax.scipy.stats import beta as jax_beta
from tensorflow_probability.substrates import jax as tfp

from ..typing import Numeric
from ..utils import jx_cast
from .crvs import ContinuousRV


class Beta(ContinuousRV):
    r"""
    Beta Random Variable

    .. math::
        X\sim \mathbf{B}(\alpha, \beta) \iff P(X=x|\alpha, \beta)=\frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha, \beta)}

        B(\alpha, \beta)=\int_{0}^{1}t^{\alpha-1}(1-t)^{\beta-1}dt
    """

    def __init__(
        self,
        alpha: Numeric | Any,
        beta: Numeric | Any,
        loc: Numeric | Any = 0.0,
        scale: Numeric | Any = 1.0,
        name: Optional[str] = None,
    ) -> None:
        shape, self._alpha, self._beta, self._loc, self._scale = jx_cast(alpha, beta, loc, scale)
        self.check_params()
        super().__init__(name=name, shape=shape)

    def check_params(self) -> None:
        assert jnp.all(self._alpha > 0.0), "alpha must be positive"
        assert jnp.all(self._beta > 0.0), "beta must be positive"
        assert jnp.all(self._scale > 0.0), "scale must be positive"

    @partial(jit, static_argnums=(0,))
    def _logpdf_x(self, x: Numeric) -> Numeric:
        return jax_beta.logpdf(
            x=x,
            a=self._alpha,
            b=self._beta,
            loc=self._loc,
            scale=self._scale,
        )

    @partial(jit, static_argnums=(0,))
    def _pdf_x(self, x: Numeric) -> Numeric:
        return jax_beta.pdf(
            x=x,
            a=self._alpha,
            b=self._beta,
            loc=self._loc,
            scale=self._scale,
        )

    @partial(jit, static_argnums=(0,))
    def _logcdf_x(self, x: Numeric) -> Numeric:
        return jax_beta.logcdf(
            x=x,
            a=self._alpha,
            b=self._beta,
            loc=self._loc,
            scale=self._scale,
        )

    @partial(jit, static_argnums=(0,))
    def _cdf_x(self, x: Numeric) -> Numeric:
        return jax_beta.cdf(
            x=x,
            a=self._alpha,
            b=self._beta,
            loc=self._loc,
            scale=self._scale,
        )

    @partial(jit, static_argnums=(0,))
    def _ppf_x(self, x: Numeric) -> Numeric:
        return tfp.math.betaincinv(
            self._alpha,
            self._beta,
            (x - self._loc) / self._scale,
        )

    def _rvs(self, shape: tuple[int, ...], key: Array) -> Array:
        return self._loc + self._scale * jax.random.beta(key=key, a=self._alpha, b=self._beta, shape=shape)

    def __repr__(self) -> str:
        string = f"Beta(alpha={self._alpha}, beta={self._beta}, loc={self._loc}, scale={self._scale}"
        if self._name is not None:
            string += f", name={self._name}"
        string += ")"
        return string
