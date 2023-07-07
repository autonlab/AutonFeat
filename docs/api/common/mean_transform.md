# Mean Transform

The mean transform computes the mean of a window. When combined with the [`SlidingWindow`](../core/fixed_window.md) abstraction, the mean transform can be used to compute the `mean` feature of a time series. The mean is defined as:

$$
\mu = \frac{1}{n} \sum_{i=1}^{n} x_i
$$

where $x_i$ is the $i$-th element of the window and $n$ is the number of elements in the window.

::: autonfeat.common.MeanTransform
      

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
tf = aft.MeanTransform()

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