# Max Transform

The min transform computes the min of a window. When combined with the [SlidingWindow](../core/fixed_window.md) abstraction, the min transform can be used to compute the `min` feature of a time series. The min is defined as:

$$
\text{min}(x) = \min_{i=1}^n x_i
$$

where $x$ is a vector of length $n$.

::: autofeat.common.MinTransform

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
tf = aft.MinTransform()

# Get featurizer
featurizer = window.use(tf)

# Get features
features = featurizer(x)

# Print features
print(window)
print(tf)
print(features)
```


If you enjoy using `AutoFeat`, please consider starring the [repository](https://github.com/autonlab/AutoFeat) ⭐️.