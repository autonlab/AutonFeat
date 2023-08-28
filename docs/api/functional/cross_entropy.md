<!-- 
Author(s): Dhruv Srikanth
Email(s): dsrikant (at) andrew (dot) cmu (dot) edu
Acknowledgements:
Copyright (c) 2023 Carnegie Mellon University, Auton Lab
This code is subject to the license terms contained in the code repo.
-->

# Cross Entropy Function

The cross-entropy function computes the cross entropy between two discrete probability distributions. The cross entropy is defined as:

$$
H(p, q) = -\sum_{x} p(x) \log q(x)
$$

where $p$ and $q$ are the two probability distributions. The cross entropy is a measure of the difference between two probability distributions. The cross entropy is zero if and only if the two distributions are identical. The cross entropy is always non-negative i.e. $H(p, q) \geq 0$.

::: autonfeat.functional.cross_entropy_tf
      

## Examples

```python
import numpy as np
import autonfeat as aft
import autonfeat.functional as F

# Random data
n_samples = 100
x1 = np.random.rand(n_samples)
x2 = np.random.rand(n_samples)

# Sliding window
ws = 10
ss = 10
window = aft.SlidingWindow(window_size=ws, step_size=ss)

# Get featurizer
featurizer = window.use(F.cross_entropy_tf)

# Get features
features = featurizer(x1, x2)

# Print features
print(features)
```

If you enjoy using [`AutonFeat`](../../index.md), please consider starring the [repository](https://github.com/autonlab/AutonFeat) ⭐️.