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

from setuptools import find_packages, setup

_current_version = '0.0.4'

with open('README.md', encoding='utf-8') as f:
    _long_description = f.read()

setup(
    name='jaxampler',
    version=_current_version,
    packages=find_packages(exclude=[
        'tests',
        'tests.*',
        'examples',
        'examples.*',
        'plots',
        'codesnap',
    ]),
    url='https://github.com/Qazalbash/jaxampler',
    license='Apache 2.0',
    author='Meesum Qazalbash',
    author_email='meesumqazalbash@gmail.com',
    description='JAX based lib for sampling statistical distributions.',
    long_description=_long_description,
    long_description_content_type='text/markdown',
    python_requires='>=3.10',
    install_requires=[
        'jaxlib>=0.4.0',
        'jax>=0.4.0',
        'tfp-nightly',
        'matplotlib>=3.8.0',
    ],
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
    ],
    zip_safe=False,
)
