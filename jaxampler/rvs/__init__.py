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

from .crvs import (Beta, Boltzmann, Cauchy, Chi2, ContinuousRV, Exponential, Gamma, Logistic, LogNormal, Normal, Pareto,
                   Rayleigh, StudentT, Triangular, TruncNormal, TruncPowerLaw, Uniform, Weibull)
from .drvs import Bernoulli, Binomial, DiscreteRV, Geometric, Poisson
from .rvs import GenericRV
