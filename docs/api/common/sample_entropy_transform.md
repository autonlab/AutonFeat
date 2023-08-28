<!-- 
Author(s): Dhruv Srikanth
Email(s): dsrikant (at) andrew (dot) cmu (dot) edu
Acknowledgements:
Copyright (c) 2023 Carnegie Mellon University, Auton Lab
This code is subject to the license terms contained in the code repo.
-->

# Sample Entropy Transform

The sample entropy transform computes the sample entropy of a window. When combined with the [`SlidingWindow`](../core/fixed_window.md) abstraction, the sample entropy transform can be used to compute the `sample entropy` feature of a time series. Sample entropy is a measure of the complexity of the signal [[1](https://en.wikipedia.org/wiki/Sample_entropy)]. It is a modification of the approximate entropy (ApEn) algorithm which is implemented [here](approx_entropy_transform.md). It is defined as:

$$
\text{Sample Entropy} = -\log\left(\frac{A}{B}\right)
$$

where $A$ is the number of matches for template vectors of length $m$ and $B$ is the number of matches for template vectors of length $m + 1$. A match is defined as a template vector $x_{m_i}$ that is close to another template vector $x_{m_j}$ in the sense that the maximum absolute difference between their corresponding scalar elements is less than or equal to a threshold $r$.

::: autonfeat.common.SampleEntropyTransform
      

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
tf = aft.SampleEntropyTransform()

# Get featurizer
featurizer = window.use(tf)

# Get features
features = featurizer(x, m=2, r=0.2)

# Print features
print(window)
print(tf)
print(features)
```

## References

[1] https://en.wikipedia.org/wiki/Sample_entropy


If you enjoy using [`AutonFeat`](../../index.md), please consider starring the [repository](https://github.com/autonlab/AutonFeat) ⭐️.