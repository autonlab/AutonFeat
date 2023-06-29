# Skewness Transform

The skew transform computes the skewness of a window. When combined with the [SlidingWindow](../core/fixed_window.md) abstraction, the skew transform can be used to compute the `skew` feature of a time series. The skewness is defined as:

$$
\gamma = \frac{m_3}{m_2^{3/2}}
$$

We use this and correct for statistical bias. The Fisher-Pearson standardized moment coefficient is defined as:

$$
\Gamma = \gamma \sqrt{\frac{N(N-1)}{N-2}}
$$

where $m_2$ and $m_3$ are the second and third central moments, respectively. They are defined as:

$$
m_2 = \frac{1}{N} \sum_{i=1}^N (x_i - \bar{x})^2
$$

$$
m_3 = \frac{1}{N} \sum_{i=1}^N (x_i - \bar{x})^3
$$

where $N$ is the number of samples in the window and $\bar{x}$ is the mean of the window.

::: autofeat.common.SkewnessTransform
      

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
tf = aft.SkewnessTransform()

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