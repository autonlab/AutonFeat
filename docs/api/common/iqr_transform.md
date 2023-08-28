<!-- 
Author(s): Dhruv Srikanth
Email(s): dsrikant (at) andrew (dot) cmu (dot) edu
Acknowledgements:
Copyright (c) 2023 Carnegie Mellon University, Auton Lab
This code is subject to the license terms contained in the code repo.
-->

# Inter-Quartile Range Transform

The inter-quartile range transform computes the inter-quartile range of the data in a sliding window. The inter-quartile range is the difference between the $75^{th}$ and $25^{th}$ percentiles of the data and can be defined as:

$$
\text{IQR} = \text{Q3} - \text{Q1}
$$

where $\text{Q1}$ and $\text{Q3}$ are the $25^{th}$ and $75^{th}$ percentiles of the data, respectively.

::: autonfeat.common.IQRTransform

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
tf = aft.IQRTransform()

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