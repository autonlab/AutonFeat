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