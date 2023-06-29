# Variance Transform

The variance transform computes the variance of a window. When combined with the [`SlidingWindow`](../core/fixed_window.md) abstraction, the variance transform can be used to compute the `var` feature of a time series. The variance is defined as:

$$
\sigma^2 = \frac{1}{n} \sum_{i=1}^{n} (x_i - \mu)^2
$$

where $x_i$ is the $i$-th element of the window, $n$ is the number of elements in the window, and $\mu$ is the mean of the window.


::: autofeat.common.VarTransform
      

## Examples

```python
import numpy as np
import autofeat as aft

# Random data
n_samples = 100
x = np.random.rand(n_samples)

# Create sliding window
ws = 10
ss = 10
window = aft.SlidingWindow(window_size=ws, step_size=ss)

# Create transform
tf = aft.VarTransform()

# Get featurizer
featurizer = window.use(tf)

# Get features
features = featurizer(x)

# Print features
print(window)
print(tf)
print(features)
```


If you enjoy using [`AutoFeat`](../../index.md), please consider starring the [repository](https://github.com/autonlab/AutoFeat) ⭐️.