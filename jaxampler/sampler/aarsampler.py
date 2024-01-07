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

from jax import Array, numpy as jnp

from .arsampler import AcceptRejectSampler
from ..rvs import ContinuousRV


class AdaptiveAcceptRejectSampler(AcceptRejectSampler):

    def __init__(self, name: str = None) -> None:
        super().__init__(name)

    def sample(self,
               target_rv: ContinuousRV,
               proposal_rv: ContinuousRV,
               scale: int = 1,
               N: int = 1,
               key: Array = None) -> Array:
        """Samples from the given random variable using the adaptive accept-reject method.

        It runs the adaptive accept-reject algorithm and returns the samples.

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
        samples = super().sample(target_rv, proposal_rv, scale, N, key)
        N_res = N - len(samples)
        while N_res != 0:
            key = self.get_key(key)
            samples = jnp.concatenate([samples, super().sample(target_rv, proposal_rv, scale, N_res, key)], axis=0)
            N_res = N - len(samples)
        return samples
