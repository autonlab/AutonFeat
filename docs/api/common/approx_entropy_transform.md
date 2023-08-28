<!-- 
Author(s): Dhruv Srikanth
Email(s): dsrikant (at) andrew (dot) cmu (dot) edu
Acknowledgements:
Copyright (c) 2023 Carnegie Mellon University, Auton Lab
This code is subject to the license terms contained in the code repo.
-->

# Approximate Entropy Transform

The approximate entropy transform computes the approximate entropy of a window. When combined with the [`SlidingWindow`](../core/fixed_window.md) abstraction, the approximate entropy transform can be used to compute the `approximate entropy` feature of a time series. It is used to quantify the amount of regularity and the unpredictability signals. Approximate entropy measures the likelihood that similar patterns of observations will not be followed by these similar patterns, therefore a time-series signal that exhibits seasonality or other kinds of repetitive patterns will have a relatively small approximate entropy whereas signals without such a repetitive nature will exhibit a high value of approximate entropy [[1](https://en.wikipedia.org/wiki/Approximate_entropy)]. It is defined as:

$$
ApEn(m, r) = \phi_{m}(r) - \phi_{m+1}(r)
$$

$$
\phi_{m}(r) = \frac{1}{N-m+1} \sum_{i=1}^{N-m+1} \ln C_m^i(r)
$$

$$
C_m^i(r) = \frac{1}{N-m+1} \sum_{j=1}^{N-m+1} \Theta(r - ||x_{i+j-1} - x_{j}||)
$$

where $m$ is the embedding dimension, $r$ is the tolerance, $N$ is the length of the signal, $x_i$ is the $i^{th}$ sample of the signal, and $\Theta$ is the Heaviside step function.

::: autonfeat.common.ApproxEntropyTransform
      

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
tf = aft.ApproxEntropyTransform()

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

[1] https://en.wikipedia.org/wiki/Approximate_entropy

If you enjoy using [`AutonFeat`](../../index.md), please consider starring the [repository](https://github.com/autonlab/AutonFeat) ⭐️.