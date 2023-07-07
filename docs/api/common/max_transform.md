# Max Transform

The max transform computes the max of a window. When combined with the [`SlidingWindow`](../core/fixed_window.md) abstraction, the max transform can be used to compute the `max` feature of a time series. The max is defined as:

$$
\text{max}(x) = \max_{i=1}^n x_i
$$

where $x$ is a vector of length $n$.

::: autonfeat.common.MaxTransform      

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
tf = aft.MaxTransform()

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