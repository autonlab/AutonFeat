# Median Function

The median function computes the median of a window. When combined with the [SlidingWindow](../core/fixed_window.md) abstraction, the median function can be used to compute the `median` feature of a time series. The median is defined as:
(write the formula as two cases for even and odd length vectors and index with i for each case)

$$
\text{median}(x) = \begin{cases}
0.5 \cdot (x_{\lfloor n/2 \rfloor} + x_{\lceil n/2 \rceil}) & \text{if $n$ is even} \\
x_{\lfloor n/2 \rfloor} & \text{if $n$ is odd}
\end{cases}
$$

where $x$ is a vector of length $n$.

::: autofeat.functional.median_tf

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
featurizer = window.use(F.median_tf)

# Get features
features = featurizer(x)

# Print features
print(features)
```


If you enjoy using `AutoFeat`, please consider starring the [repository](https://github.com/autonlab/AutoFeat) ⭐️.