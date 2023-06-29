# Mean Function

The mean function computes the mean of a signal. Mean is often used as a summary statistic for a signal. Using the [`SlidingWindow`](../core/fixed_window.md) abstraction, the mean can be computed over a sliding window of the signal to be produce a set of features that can be used for a downstream task.

::: autofeat.functional.mean
      

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
featurizer = window.use(F.mean)

# Get features
features = featurizer(x)

# Print features
print(features)
```


If you enjoy using [`AutoFeat`](../../index.md), please consider starring the [repository](https://github.com/autonlab/AutoFeat) ⭐️.