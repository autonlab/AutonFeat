<!-- 
MIT License

Copyright (c) 2023 Carnegie Mellon University, Auton Lab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
-->

# Introduction

[`AutonFeat`](../index.md) is a automatic featurization library for time-series data. It is designed to be used in conjunction with machine learning libraries such as [scikit-learn](https://scikit-learn.org/stable/) and [PyTorch](https://pytorch.org/). It is built on design principles similar to that of [PyTorch](https://pytorch.org/) offering a simple and flexible API.

More about the package can be found in [here](../index.md). If you are interested in contributing to the project, please see the [contributing guide](../community/contributing.md). If you enjoy using [`AutonFeat`](../index.md), please consider starring the [repository](https://github.com/autonlab/AutonFeat) ⭐️.

## Why AutonFeat?

What sets this package apart from the many packages already available to researchers and practitioners? We believe that [`AutonFeat`](../index.md) is unique in the following ways:

- **Automatic**: [`AutonFeat`](../index.md) is designed to be used with minimal user input. The user only needs to specify the input data and the featurization method(s). The package will automatically featurize the data and return a set of features that can be used for a downstream task.

- **Simple & Interpretable**: [`AutonFeat`](../index.md) is designed to be interpretable. The user can understand the featurization process and the resulting features through good documentation.

- **Flexible & Extensible**: [`AutonFeat`](../index.md) is designed to be flexible and extensible. The user can easily extend the package to include custom featurization functions.

- **Fast**: [`AutonFeat`](../index.md) is designed to be fast enough to be used in production. Our benchmarks prove the utility of our design choices against existing implementations and packages with *truly* multi-threaded support. Operations are vectorized where possible and parallelized where necessary. We utilize [numba](https://numba.pydata.org/) and [numpy](https://numpy.org/) to speed up the featurization process, escaping Python's Global Interpreter Lock (GIL).

## Jump In

To get started with [`AutonFeat`](../index.md), we recommend reading the [quickstart guide](installation.md) and following the tutorials we've provided [here](../tutorials/tutorials.md). If you are interested in the API, you can find the documentation [here](../api/api.md).

If you enjoy using [`AutonFeat`](../index.md), please consider starring the [repository](https://github.com/autonlab/AutonFeat) ⭐️.