# Quantile Transform

The quantile transform computes the q-th quantile of the data in the sliding window. The quantile is computed using the [numpy.quantile](https://numpy.org/doc/stable/reference/generated/numpy.quantile.html) function. The transform can be combined with the [SlidingWindow](../core/fixed_window.md) to compute the quantile of the data in a sliding window. We can use this transform to compute the median of the data in a sliding window by setting `q=0.5`.

::: autofeat.common.QuantileTransform

## Examples

### 25th percentile

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
tf = aft.QuantileTransform()

# Get featurizer
featurizer = window.use(tf)

# Get features
features = featurizer(x, q=0.25)

# Print features
print(window)
print(tf)
print(features)
```

### Median

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
tf = aft.QuantileTransform()

# Get featurizer
featurizer = window.use(tf)

# Get features
features = featurizer(x, q=0.5)

# Print features
print(window)
print(tf)
print(features)
```


If you enjoy using `AutoFeat`, please consider starring the [repository](https://github.com/autonlab/AutoFeat) ⭐️.