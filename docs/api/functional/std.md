# Standard Deviation Function

The standard deviation function computes the standard deviation of a window. When combined with the [SlidingWindow](../core/fixed_window.md) abstraction, the standard deviation function can be used to compute the `std` feature of a time series. The standard deviation is defined as:

$$
\sigma = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (x_i - \mu)^2}
$$

where $x_i$ is the $i$-th element of the window, $n$ is the number of elements in the window, and $\mu$ is the mean of the window.

::: autofeat.functional.std_tf
      

## Examples

```python
import numpy as np
import autofeat as aft
import autofeat.functional as F

# Random data
n_samples = 100
x = np.random.rand(n_samples)

# Create sliding window
ws = 10
ss = 10
window = aft.SlidingWindow(window_size=ws, step_size=ss)

# Get featurizer
featurizer = window.use(F.std_tf)

# Get features
features = featurizer(x)

# Print features
print(features)
```


If you enjoy using `AutoFeat`, please consider starring the [repository](https://github.com/autonlab/AutoFeat) ⭐️.