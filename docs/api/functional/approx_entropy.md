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

# Approximate Entropy Function

The approximate entropy function computes the approximate entropy of a window. When combined with the [`SlidingWindow`](../core/fixed_window.md) abstraction, the approximate entropy function can be used to compute the `approximate entropy` feature of a time series. It is used to quantify the amount of regularity and the unpredictability signals. Approximate entropy measures the likelihood that similar patterns of observations will not be followed by these similar patterns, therefore a time-series signal that exhibits seasonality or other kinds of repetitive patterns will have a relatively small approximate entropy whereas signals without such a repetitive nature will exhibit a high value of approximate entropy [[1](https://en.wikipedia.org/wiki/Approximate_entropy)]. It is defined as:

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

::: autonfeat.functional.approx_entropy_tf
      

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
featurizer = window.use(F.approx_entropy_tf)

# Get features
features = featurizer(x, m=2, r=0.2)

# Print features
print(features)
```

## References

[1] https://en.wikipedia.org/wiki/Approximate_entropy

If you enjoy using [`AutonFeat`](../../index.md), please consider starring the [repository](https://github.com/autonlab/AutonFeat) ⭐️.