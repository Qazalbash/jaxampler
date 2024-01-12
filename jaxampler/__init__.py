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


import importlib.metadata

from . import _typing as typing
from ._montecarlo import (
    Integration as Integration,
    MonteCarloBoxIntegration as MonteCarloBoxIntegration,
    MonteCarloGenericIntegration as MonteCarloGenericIntegration,
)
from ._rvs import (
    Bernoulli as Bernoulli,
    Beta as Beta,
    Binomial as Binomial,
    Boltzmann as Boltzmann,
    Cauchy as Cauchy,
    Chi2 as Chi2,
    ContinuousRV as ContinuousRV,
    DiscreteRV as DiscreteRV,
    Exponential as Exponential,
    Gamma as Gamma,
    GenericRV as GenericRV,
    Geometric as Geometric,
    Logistic as Logistic,
    LogNormal as LogNormal,
    Normal as Normal,
    Pareto as Pareto,
    Poisson as Poisson,
    Rayleigh as Rayleigh,
    StudentT as StudentT,
    Triangular as Triangular,
    TruncNormal as TruncNormal,
    TruncPowerLaw as TruncPowerLaw,
    Uniform as Uniform,
    Weibull as Weibull,
)
from ._sampler import (
    AcceptRejectSampler as AcceptRejectSampler,
    AdaptiveAcceptRejectSampler as AdaptiveAcceptRejectSampler,
    ImportanceSampler as ImportanceSampler,
    InverseTransformSampler as InverseTransformSampler,
    MetropolisHastingSampler as MetropolisHastingSampler,
    Sampler as Sampler,
)
from ._utils import jx_cast as jx_cast, nCr as nCr, nPr as nPr


__version__ = importlib.metadata.version("jaxampler")
