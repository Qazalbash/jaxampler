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

from setuptools import find_packages, setup

with open('README.md', encoding='utf-8') as f:
    _long_description = f.read()

setup(
    name='jaxampler',
    version='0.0.1',
    packages=find_packages(),
    url='https://github.com/Qazalbash/jaxampler',
    license='Apache 2.0',
    author='Meesum Qazalbash',
    author_email='meesumqazalbash@gmail.com',
    description='jaxampler is a JAX based lib for sampling statistical distributions.',
    long_description=_long_description,
    long_description_content_type='text/markdown',
    python_requires='>=3.9',
    install_requires=['jaxlib>=0.4.23', 'jax>=0.4.23', 'tfp-nightly', 'matplotlib>=3.8.2'],
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    zip_safe=False,
)
