# Inter-Quartile Range Function

The inter-quartile range function computes the inter-quartile range of the data in a sliding window. The inter-quartile range is the difference between the $75^{th}$ and $25^{th}$ percentiles of the data and can be defined as:

$$
\text{IQR} = \text{Q3} - \text{Q1}
$$

where $\text{Q1}$ and $\text{Q3}$ are the $25^{th}$ and $75^{th}$ percentiles of the data, respectively.

::: autofeat.functional.iqr_tf

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
featurizer = window.use(F.iqr_tf)

# Get features
features = featurizer(x)

# Print features
print(features)
```

If you enjoy using `AutoFeat`, please consider starring the [repository](https://github.com/autonlab/AutoFeat) ⭐️.