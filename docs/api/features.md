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

# Features

Feature extractors are used, as the name suggests, to extract features from a signal. [`AutonFeat`](../index.md) provides a wide range of feature extractors that can be used to extract features from a signal. The following sections describe the various feature extractors that are available in [`AutonFeat`](../index.md).

## Domain Agnostic

Domain agnostic features are applicable to *most* signals irrespective of the domain.

### Summary Statistics

| Feature | Description | Endpoint |
| --- | --- | --- |
| Max | Maximum value of the signal | [`MaxTransform`](common/max_transform.md) |
| Min | Minimum value of the signal | [`MinTransform`](common/min_transform.md) |
| Mean | Mean of the signal | [`MeanTransform`](common/mean_transform.md) |
| Median | Median of the signal | [`MedianTransform`](common/median_transform.md) |
| Standard Deviation | Standard deviation of the signal | [`StdTransform`](common/std_transform.md) |
| Variance | Variance of the signal | [`VarTransform`](common/var_transform.md) |
| Quantile | Quantile of the signal | [`QuantileTransform`](common/quantile_transform.md) |
| Range | Range of the signal | [`RangeTransform`](common/range_transform.md) |
| IQR | Interquartile range of the signal | [`IQRTransform`](common/iqr_transform.md) |
| N Valid | Number of valid values in the signal | [`NValidTransform`](common/n_valid_transform.md) |
| Skewness | Skewness of the signal | [`SkewnessTransform`](common/skewness_transform.md) |
| Kurtosis | Kurtosis of the signal | [`KurtosisTransform`](common/kurtosis_transform.md) |

### Data Sparsity Measures

| Feature | Description | Endpoint |
| --- | --- | --- |
| Data Density | Ratio of valid values to window size | [`DataDensityTransform`](common/data_density_transform.md) |
| Data Sparsity | Ratio of missing values to window size | [`DataSparsityTransform`](common/data_sparsity_transform.md) |

### Information Theoretic Measures

| Feature | Description | Endpoint |
| --- | --- | --- |
| Shannon Entropy | Shannon entropy of the signal | [`EntropyTransform`](common/entropy_transform.md) |
| KL Divergence | KL divergence of the signal with another distribution | [`EntropyTransform`](common/entropy_transform.md) |
| Sample Entropy | Sample entropy of the signal | [`SampleEntropyTransform`](common/sample_entropy_transform.md) |
| Cross Entropy | Cross entropy of the signal with another distribution | [`CrossEntropyTransform`](common/cross_entropy_transform.md) |

## Domain Specific

Domain expertise *almost always* helps in extracting better features. [`AutonFeat`](../index.md) provides a wide range of domain specific features that can be used to extract features from a signal. The following sections describe the various domain specific feature extractors that are available in [`AutonFeat`](../index.md).

### Biomedical, and Physiological Signals

***(Coming Soon)***

| Feature | Description | Endpoint |
| --- | --- | --- |


## Functional Form

A functional form for each of the transforms above is also provided for convenience. Check out the **`autonfeat.functional`** sub-module for more details.

## Custom Featurizers

It is possible to design custom features while extending the functionality of [`AutonFeat`](../index.md). The [`Transform`](core/transform.md) class is an abstraction representing a featurizer. When defining a custom featurizer, one can utilize the efficiency of the [`SlidingWindow`](core/fixed_window.md) abstraction by inhering defining the featurizer to inherit from [`Transform`](core/transform.md). The following example demonstrates how to create a custom feature that computes the mean of the signal.

```python
import numpy as np
from typing import Callable, Union
from autonfeat.core import Transform

class MeanTransform(Transform):
    def __init__(self, name: str = "Mean") -> None:
        super().__init__(name=name)

    def __call__(self, signal_window: np.ndarray, where: Callable[[Union[int, float, np.int_, np.float_]], Union[bool, np.bool_]] = lambda x: not np.isnan(x)) -> Union[np.float_, np.int_]:
        where_fn = np.vectorize(pyfunc=where)
        return np.mean(x, where=where_fn(x))
```

This can then be passed as the featurizer to be applied at every sliding window interval as such - 

```python
import autonfeat as aft

# Random data
n_samples = 100
x = np.random.rand(n_samples)

# Create sliding window
ws = 10
ss = 10
window = aft.SlidingWindow(window_size=ws, step_size=ss)

# Create transform
tf = MeanTransform()

# Get featurizer
featurizer = window.use(tf)

# Get features
features = featurizer(x)

# Print features
print(window)
print(tf)
print(features)
```

See [this](../tutorials/tutorials.md) for more examples on how to use feature extractors in [`AutonFeat`](../index.md).


If you enjoy using [`AutonFeat`](../index.md), please consider starring the [repository](https://github.com/autonlab/AutonFeat) ⭐️.