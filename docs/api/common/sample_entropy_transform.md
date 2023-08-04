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