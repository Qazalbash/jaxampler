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

import jax
from jax import Array

from ..rvs import ContinuousRV
from .sampler import Sampler


class AcceptRejectSampler(Sampler):
    """AcceptRejectSampler is a sampler that uses the accept-reject method
    to sample from a random variable."""

    def __init__(self) -> None:
        """Initializes an AcceptRejectSampler object."""
        super().__init__()

    def sample(self,
               target_rv: ContinuousRV,
               proposal_rv: ContinuousRV,
               scale: int = 1,
               N: int = 1,
               key: Array = None) -> Array:
        """Samples from the given random variable using the accept-reject method.

        It runs the accept-reject algorithm and returns the samples.

        Parameters
        ----------
        target_rv : ContinuousRV
            The random variable to sample from.
        proposal_rv : ContinuousRV
            The proposal random variable.
        scale : int, optional
            Scaler to cover target distribution by proposal distribution, by default 1
        N : int, optional
            Number of samples, by default 1
        key : Array, optional
            The key to use for sampling, by default None

        Returns
        -------
        Array
            The samples.
        """
        self.check_rv(target_rv)
        self.check_rv(proposal_rv)

        if key is None:
            key = self.get_key(key)

        V = proposal_rv.rvs((1, N), key)

        pdf = target_rv.pdf_v(*V)

        key = self.get_key(key)
        U_scaled = jax.random.uniform(
            key,
            shape=(N,),
            minval=0.0,
            maxval=scale * proposal_rv.pdf_v(*V),
        )

        accept = U_scaled <= pdf
        samples = (V.T)[accept]
        samples = samples.flatten()
        return samples
