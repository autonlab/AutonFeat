# Kurtosis Function

The kurtosis function computes the kurtosis of a window. When combined with the [SlidingWindow](../core/fixed_window.md) abstraction, the kurtosis function can be used to compute the `kurtosis` feature of a time series. The kurtosis is defined as:

$$
\kappa = \begin{cases}
\frac{m_4}{m_2^2} - 3 & \text{Fisher} \\
\frac{m_4}{m_2^2} & \text{Pearson}
\end{cases}
$$

where $m_2$ and $m_4$ are the second and fourth central moments, respectively. They are defined as:

$$
m_2 = \frac{1}{N} \sum_{i=1}^N (x_i - \bar{x})^2
$$

$$
m_4 = \frac{1}{N} \sum_{i=1}^N (x_i - \bar{x})^4
$$

where $N$ is the number of samples in the window and $\bar{x}$ is the mean of the window.

::: autofeat.functional.kurtosis_tf
      

## Examples

### Fisher Kurtosis

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
featurizer = window.use(F.kurtosis_tf)

# Get features
features = featurizer(x)

# Print features
print(features)
```

### Pearson Kurtosis

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
featurizer = window.use(F.kurtosis_tf)

# Get features
features = featurizer(x, fisher=False)

# Print features
print(features)
```

If you enjoy using [`AutoFeat`](../../index.md), please consider starring the [repository](https://github.com/autonlab/AutoFeat) ⭐️.