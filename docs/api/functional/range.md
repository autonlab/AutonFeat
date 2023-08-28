<!-- 
Author(s): Dhruv Srikanth
Email(s): dsrikant (at) andrew (dot) cmu (dot) edu
Acknowledgements:
Copyright (c) 2023 Carnegie Mellon University, Auton Lab
This code is subject to the license terms contained in the code repo.
-->

# Range Function

The range function computes the range of the data in the sliding window. When paired with the [`SlidingWindow`](../core/fixed_window.md) abstraction, one can compute the range over a sliding window across a time series. The range is computed as the difference between the maximum and minimum values in the window and can be defined as:

$$
\text{range} = \max(x) - \min(x)
$$

where $x$ is the data in the sliding window.

::: autonfeat.functional.range_tf

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
featurizer = window.use(F.range_tf)

# Get features
features = featurizer(x)

# Print features
print(features)
```


If you enjoy using [`AutonFeat`](../../index.md), please consider starring the [repository](https://github.com/autonlab/AutonFeat) ⭐️.