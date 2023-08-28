<!-- 
Author(s): Dhruv Srikanth
Email(s): dsrikant (at) andrew (dot) cmu (dot) edu
Acknowledgements:
Copyright (c) 2023 Carnegie Mellon University, Auton Lab
This code is subject to the license terms contained in the code repo.
-->

# Data Density Transform

The data density transform computes the ratio of valid values in a sliding window to the total number of values in the window. See [`NValidTransform`](n_valid_transform.md) for more details on how valid values are computed. It can be coupled with the [`SlidingWindow`](../core/fixed_window.md) abstraction to compute the `data density` feature of a time series. It can be defined as:

$$
\text{density} = \frac{N_{valid}}{N_{total}}
$$

where $N_{valid}$ is the number of valid values in a window $W$ and $N_{total}$ is the total number of values in $W$.


::: autonfeat.common.DataDensityTransform
      

## Examples

```python
import numpy as np
import autonfeat as aft

# Random data
n_samples = 100
x = np.random.rand(n_samples)

# Create sliding window
ws = 10
ss = 10
window = aft.SlidingWindow(window_size=ws, step_size=ss)

# Create transform
tf = aft.DataDensityTransform()

# Get featurizer
featurizer = window.use(tf)

# Get features
features = featurizer(x)

# Print features
print(window)
print(tf)
print(features)
```

If you enjoy using [`AutonFeat`](../../index.md), please consider starring the [repository](https://github.com/autonlab/AutonFeat) ⭐️.