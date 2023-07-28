# MIT License

# Copyright (c) 2023 Carnegie Mellon University, Auton Lab

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.md')) as f:
    long_description = f.read()

user_requirements = [
    requirement.strip()
    for requirement in open(
        path.join(here, 'requirements.txt')
    ).readlines()
]

setup(
    name='autonfeat',
    version='0.1.0',
    author=[
        'Dhruv Srikanth',
        'Auton Lab'
    ],
    author_email='dsrikant@andrew.cmu.edu',
    description='A High Performance Library for Time-Series Featurization.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='MIT',
    url='https://autonlab.github.io/AutoFeat',
    keywords=[
        'artificial intelligence',
        'machine learning',
        'data science',
        'data analysis',
        'time-series',
        'featurization',
        'forecasting',
        'high-performance computing',
    ],
    packages=find_packages(where='autonfeat'),
    package_dir={'': 'autonfeat'},
    package_data={'': ['autonfeat/utils/datasets/data/*.csv']},
    install_requires=user_requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
