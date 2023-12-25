# Copyright 2023 The JAXampler Authors

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

    def __init__(self) -> None:
        super().__init__()

    def sample(self,
               target_rv: ContinuousRV,
               proposal_rv: ContinuousRV,
               scale: int = 1,
               N: int = 1,
               key: Array = None) -> Array:
        self.check_rv(target_rv)
        self.check_rv(proposal_rv)

        if key is None:
            key = self.get_key(key)

        V = proposal_rv.rvs(N, key)

        pdf_ratio = target_rv.pdf(*(V.T))

        key = self.get_key(key)
        U_scaled = jax.random.uniform(
            key,
            shape=(N,),
            minval=0.0,
            maxval=scale * proposal_rv.pdf(*(V.T)),
        )

        accept = U_scaled <= pdf_ratio
        samples = V[accept]

        return samples
