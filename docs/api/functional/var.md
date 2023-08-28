<!-- 
Author(s): Dhruv Srikanth
Email(s): dsrikant (at) andrew (dot) cmu (dot) edu
Acknowledgements:
Copyright (c) 2023 Carnegie Mellon University, Auton Lab
This code is subject to the license terms contained in the code repo.
-->

# Variance Function

The variance function computes the variance of a window. When combined with the [`SlidingWindow`](../core/fixed_window.md) abstraction, the variance function can be used to compute the `var` feature of a time series. The variance is defined as:

$$
\sigma^2 = \frac{1}{n} \sum_{i=1}^{n} (x_i - \mu)^2
$$

where $x_i$ is the $i$-th element of the window, $n$ is the number of elements in the window, and $\mu$ is the mean of the window.


::: autonfeat.functional.var_tf
      

## Examples

```python
import numpy as np
import autonfeat as aft
import autonfeat.functional as F

# Random data
n_samples = 100
x = np.random.rand(n_samples)

# Create sliding window
ws = 10
ss = 10
window = aft.SlidingWindow(window_size=ws, step_size=ss)

# Get featurizer
featurizer = window.use(F.var_tf)

# Get features
features = featurizer(x)

# Print features
print(features)
```


If you enjoy using [`AutonFeat`](../../index.md), please consider starring the [repository](https://github.com/autonlab/AutonFeat) ⭐️.