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

# Entropy Transform

The entropy transform computes the entropy of a distribution. The entropy is a measure of the uncertainty of a random variable. The entropy of a distribution is defined as:

$$
H(X) = -\sum_{i=1}^{n} p(x_i) \log p(x_i)
$$

where $p(x_i)$ is the probability of the $i$-th outcome. The entropy is maximized when all outcomes are equally likely. The entropy is zero when the distribution is deterministic.

We can use the entropy transform to compute the entropy of a single discrete probability distribution using *Shannon Entropy*. We can also use the entropy transform to compute the relative entropy between two discrete probability distributions. This is also called the *Kullback-Leibler (KL) divergence*. This is defined as:

$$
D_{KL}(p||q) = H(p, q) = \sum_{x} p(x) \log \frac{p(x)}{q(x)}
$$

where $p$ and $q$ are the two probability distributions. The relative entropy is zero if and only if the two distributions are identical. The relative entropy is always non-negative.

::: autonfeat.common.EntropyTransform
      

## Examples

### Shannon Entropy

```python
import numpy as np
import autonfeat as aft

# Random data
n_samples = 100
x = np.random.randint(0, 10, n_samples)

# Sliding window
ws = 10
ss = 10
window = aft.SlidingWindow(window_size=ws, step_size=ss)

# Create transform
tf = aft.EntropyTransform()

# Get featurizer
featurizer = window.use(tf)

# Get features
features = featurizer(x)

# Print features
print(features)
```

### KL Divergence

```python
import numpy as np
import autonfeat as aft

# Random data
n_samples = 100
x1 = np.random.rand(n_samples)
x2 = np.random.rand(n_samples)

# Sliding window
ws = 10
ss = 10
window = aft.SlidingWindow(window_size=ws, step_size=ss)

# Create transform
tf = aft.EntropyTransform()

# Get featurizer
featurizer = window.use(tf)

# Get features
features = featurizer(x1, x2)

# Print features
print(features)
```

If you enjoy using [`AutonFeat`](../../index.md), please consider starring the [repository](https://github.com/autonlab/AutonFeat) ⭐️.