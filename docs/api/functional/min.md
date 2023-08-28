<!-- 
Author(s): Dhruv Srikanth
Email(s): dsrikant (at) andrew (dot) cmu (dot) edu
Acknowledgements:
Copyright (c) 2023 Carnegie Mellon University, Auton Lab
This code is subject to the license terms contained in the code repo.
-->

# Max Function

The min function computes the min of a window. When combined with the [`SlidingWindow`](../core/fixed_window.md) abstraction, the min function can be used to compute the `min` feature of a time series. The min is defined as:

$$
\text{min}(x) = \min_{i=1}^n x_i
$$

where $x$ is a vector of length $n$.

::: autonfeat.functional.min_tf

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
featurizer = window.use(F.min_tf)

# Get features
features = featurizer(x)

# Print features
print(features)
```


If you enjoy using [`AutonFeat`](../../index.md), please consider starring the [repository](https://github.com/autonlab/AutonFeat) ⭐️.