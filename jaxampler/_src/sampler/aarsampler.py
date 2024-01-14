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

from typing import Optional

from jax import Array, numpy as jnp

from ..rvs.crvs import ContinuousRV
from .arsampler import AcceptRejectSampler


class AdaptiveAcceptRejectSampler(AcceptRejectSampler):
    def __init__(self, name: Optional[str] = None) -> None:
        super().__init__(name)

    def sample(self, *args, **kwargs) -> Array:
        """Samples from the given random variable using the adaptive accept-reject method.

        It runs the adaptive accept-reject algorithm and returns the samples.

        Parameters
        ----------
        target_rv : ContinuousRV
            The random variable to sample from.
        proposal_rv : ContinuousRV
            The proposal random variable.
        scale : float, optional
            Scaler to cover target distribution by proposal distribution, by default 1.0
        N : int
            Number of samples
        key : Array, optional
            The key to use for sampling, by default None

        Returns
        -------
        Array
            The samples.
        """
        target_rv: Optional[ContinuousRV] = kwargs.get("target_rv", None)
        proposal_rv: Optional[ContinuousRV] = kwargs.get("proposal_rv", None)
        N: Optional[int] = kwargs.get("N", None)

        assert target_rv is not None, "target_rv is None"
        assert proposal_rv is not None, "proposal_rv is None"
        assert N is not None, "N is None"

        scale: float = kwargs.get("scale", 1.0)
        key: Optional[Array] = kwargs.get("key", None)

        samples = super().sample(target_rv, proposal_rv, scale, N, key)
        N_res = N - len(samples)
        while N_res != 0:
            key = self.get_key(key)
            samples = jnp.concatenate(
                [samples, super().sample(target_rv, proposal_rv, scale, N_res, key)],
                axis=0,
            )
            N_res = N - len(samples)
        return samples
