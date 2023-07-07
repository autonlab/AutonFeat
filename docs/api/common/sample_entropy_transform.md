# Sample Entropy Transform

The sample entropy transform computes the sample entropy of a window. When combined with the [`SlidingWindow`](../core/fixed_window.md) abstraction, the mean transform can be used to compute the `sample entropy` feature of a time series. Sample entropy is a measure of the complexity of the signal. It is a modification of the approximate entropy (ApEn) algorithm. It is defined as:

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
features = featurizer(x)

# Print features
print(window)
print(tf)
print(features)
```


If you enjoy using [`AutonFeat`](../../index.md), please consider starring the [repository](https://github.com/autonlab/AutonFeat) ⭐️.