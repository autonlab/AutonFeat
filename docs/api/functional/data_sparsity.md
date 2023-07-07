# Data Sparsity Function

The data sparsity function computes the ratio of invalid values in a sliding window to the total number of values in the window. See [`n-valid`](n_valid.md) for more details on how valid and invalid values are computed. It can be coupled with the [`SlidingWindow`](../core/fixed_window.md) abstraction to compute the `data sparsity` feature of a time series. It can be defined as:

$$
\text{sparsity} = \frac{N_{invalid}}{N_{total}}
$$

where $N_{invalid}$ is the number of invalid values in a window $W$ and $N_{total}$ is the total number of values in $W$.


::: autonfeat.functional.data_sparsity_tf
      

## Examples

```python
import numpy as np
import autonfeat as aft
import autonfeat.functional as F

# Random data
n_samples = 100
x = np.random.rand(n_samples)

# Create sliding window
ws = 10
ss = 10
window = aft.SlidingWindow(window_size=ws, step_size=ss)

# Get featurizer
featurizer = window.use(F.data_sparsity_tf)

# Get features
features = featurizer(x)

# Print features
print(features)
```

If you enjoy using [`AutonFeat`](../../index.md), please consider starring the [repository](https://github.com/autonlab/AutonFeat) ⭐️.