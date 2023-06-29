# Quantile Function

The quantile function computes the q-th quantile of the data in the sliding window. The quantile is computed using the [numpy.quantile](https://numpy.org/doc/stable/reference/generated/numpy.quantile.html) function. The function can be combined with the [SlidingWindow](../core/fixed_window.md) to compute the quantile of the data in a sliding window. We can use this function to compute the median of the data in a sliding window by setting `q=0.5`.

::: autofeat.functional.quantile_tf

## Examples

### 25th percentile

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
featurizer = window.use(F.quantile_tf)

# Get features
features = featurizer(x, q=0.25)

# Print features
print(features)
```

### Median

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
featurizer = window.use(F.quantile_tf)

# Get features
features = featurizer(x, q=0.5)

# Print features
print(features)
```


If you enjoy using `AutoFeat`, please consider starring the [repository](https://github.com/autonlab/AutoFeat) ⭐️.