# Introduction

`AutoFeat` is a automatic featurization library for time-series data. It is designed to be used in conjunction with machine learning libraries such as [scikit-learn](https://scikit-learn.org/stable/) and [PyTorch](https://pytorch.org/). `AutoFeat` is built on design principles similar to that of [PyTorch](https://pytorch.org/).

More about the package can be found in [here](../index.md). If you are interested in contributing to the project, please see the [contributing guide](../community/contributing.md). If you enjoy using `AutoFeat`, please consider starring the [repository](https://github.com/autonlab/AutoFeat) ⭐️.

## Why AutoFeat?

What sets this package apart from the many packages already available to researchers and practitioners? We believe that `AutoFeat` is unique in the following ways:

- **Automatic**: `AutoFeat` is designed to be used with minimal user input. The user only needs to specify the input data and the featurization method(s). The package will automatically featurize the data and return a set of features that can be used for a downstream task.

- **Simple & Interpretable**: `AutoFeat` is designed to be interpretable. The user can understand the featurization process and the resulting features through good documentation.

- **Flexible & Extensible**: `AutoFeat` is designed to be flexible and extensible. The user can easily extend the package to include custom featurization functions.

- **Fast**: `AutoFeat` is designed to be fast enough to be used in production. Our benchmarks prove the utility of our design choices against existing implementations and packages with *truly* multi-threaded support. Operations are vectorized where possible and parallelized where necessary. We utilize [numba](https://numba.pydata.org/) and [numpy](https://numpy.org/) to speed up the featurization process, escaping Python's Global Interpreter Lock (GIL) where possible.

## Jump In

To get started with `AutoFeat`, we recommend reading the [quickstart guide](installation.md) and following the tutorials we've provided [here](../tutorials/tutorials.md). If you are interested in the API, you can find the documentation [here](../api/api.md).

If you enjoy using `AutoFeat`, please consider starring the [repository](https://github.com/autonlab/AutoFeat) ⭐️.