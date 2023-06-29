# Max Function

The max function computes the max of a window. When combined with the [SlidingWindow](../core/fixed_window.md) abstraction, the max function can be used to compute the `max` feature of a time series. The max is defined as:

$$
\text{max}(x) = \max_{i=1}^n x_i
$$

where $x$ is a vector of length $n$.

::: autofeat.functional.max_tf      

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
featurizer = window.use(F.max_tf)

# Get features
features = featurizer(x)

# Print features
print(features)
```

If you enjoy using `AutoFeat`, please consider starring the [repository](https://github.com/autonlab/AutoFeat) ⭐️.
