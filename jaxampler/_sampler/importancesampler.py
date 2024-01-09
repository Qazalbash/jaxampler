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

from typing import Callable, Optional

from jax import Array, vmap

from .._rvs import ContinuousRV
from .sampler import Sampler


class ImportanceSampler(Sampler):
    """ImportanceSampler is a sampler that uses the importance sampling method
    to sample from a random variable."""

    def __init__(self, name: Optional[str] = None) -> None:
        super().__init__(name)

    def sample(
        self,
        h: Callable,
        p: ContinuousRV,
        q: ContinuousRV,
        *args,
        N: int = 1,
        key: Optional[Array] = None,
        **kwargs,
    ) -> Array:
        """Samples from the given random variable using the importance sampling method.

        It runs the importance sampling algorithm and returns the samples.

        Parameters
        ----------
        h : Callable
            function to be integrated
        p : ContinuousRV
            target distribution
        q : ContinuousRV
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
        if key is None:
            key = self.get_key()
        q_rv = q.rvs(shape=(N,), key=key)
        p_theta = p.pdf_v(q_rv)
        q_phi = q.pdf_v(q_rv)
        hx = vmap(h)(q_rv)
        w = p_theta / q_phi
        normalization_constant = w.mean()
        expected_value = (w * hx).mean() / normalization_constant
        return expected_value
