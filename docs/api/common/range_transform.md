# Range Transform

The range transform computes the range of the data in the sliding window. When paired with the [SlidingWindow](../core/fixed_window.md) abstraction, one can compute the range over a sliding window across a time series. The range is computed as the difference between the maximum and minimum values in the window and can be defined as:

$$
\text{range} = \max(x) - \min(x)
$$

where $x$ is the data in the sliding window.

::: autofeat.common.RangeTransform

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
tf = aft.RangeTransform()

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