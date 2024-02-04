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

from typing import Callable, Optional

from jax import Array, vmap

from ..rvs.rvs import RandomVariable
from .sampler import Sampler


class ImportanceSampler(Sampler):
    """ImportanceSampler is a sampler that uses the importance sampling method
    to sample from a random variable."""

    def __init__(self, name: Optional[str] = None) -> None:
        super().__init__(name)

    def sample(self, *args, **kwargs) -> Array:
        """Samples from the given random variable using the importance sampling method.

        It runs the importance sampling algorithm and returns the samples.

        Parameters
        ----------
        h : Callable
            function to be integrated
        p : RandomVariable
            target distribution
        q : RandomVariable
            proxy distribution
        N : int, optional
            Number of samples, by default 1
        key : Array, optional
            JAX PRNGKey, by default None

        Returns
        -------
        Array
            Samples from the target distribution
        """
        h: Optional[Callable] = kwargs.get("h", None)
        p: Optional[RandomVariable] = kwargs.get("p", None)
        q: Optional[RandomVariable] = kwargs.get("q", None)
        N: Optional[int] = kwargs.get("N", None)

        assert h is not None, "h is None"
        assert p is not None, "p is None"
        assert q is not None, "q is None"
        assert N is not None, "N is None"

        seed: Optional[int] = kwargs.get("seed", None)

        q_rv = q.rvs(shape=(N,), seed=seed)
        p_theta = p._pdf_v(q_rv)
        q_phi = q._pdf_v(q_rv)
        hx = vmap(h)(q_rv)
        w = p_theta / q_phi
        normalization_constant = w.mean()
        expected_value = (w * hx).mean() / normalization_constant
        return expected_value
