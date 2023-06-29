# N-Valid Transform

Compute the number of valid measurements in a sliding window. A valid measurement by default is defined as a measurement that is not `np.nan`, however this can be altered by passing a *validity* function to the argument `where`. The *validity* function should take a single argument, the measurement, and return `True` if the measurement is valid, and `False` otherwise. The transform can be defined as:

$$
\mathbb{1}_{\text{valid}}(x_i) = \begin{cases}
1 & \text{if } x_i \text{ is valid} \\
0 & \text{otherwise}
\end{cases}
$$

where $x_i$ is the $i$-th measurement in the sliding window.

$$
\text{NValid} = \sum_{i=1}^n \mathbb{1}_{\text{valid}}(x_i)
$$

where $n$ is the number of measurements in the sliding window.

::: autofeat.common.NValidTransform
      

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
tf = aft.NValidTransform()

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