# Data Sparsity Transform

The data sparsity transform computes the ratio of invalid values in a sliding window to the total number of values in the window. See [NValidTransform](n_valid_transform.md) for more details on how invalid values are computed. It can be coupled with the [SlidingWindow](../core/fixed_window.md) abstraction to compute the `data_sparsity` feature of a time series. It can be defined as:

$$
\text{sparsity} = \frac{N_{invalid}}{N_{total}}
$$

where $N_{invalid}$ is the number of invalid values in a window $W$ and $N_{total}$ is the total number of values in $W$.


::: autofeat.common.DataSparsityTransform
      

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
tf = aft.DataSparsityTransform()

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