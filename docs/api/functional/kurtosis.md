<!-- 
MIT License

Copyright (c) 2023 Carnegie Mellon University, Auton Lab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
-->

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

::: autonfeat.functional.kurtosis_tf
      

## Examples

### Fisher Kurtosis

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
featurizer = window.use(F.kurtosis_tf)

# Get features
features = featurizer(x)

# Print features
print(features)
```

### Pearson Kurtosis

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
featurizer = window.use(F.kurtosis_tf)

# Get features
features = featurizer(x, fisher=False)

# Print features
print(features)
```

If you enjoy using [`AutonFeat`](../../index.md), please consider starring the [repository](https://github.com/autonlab/AutonFeat) ⭐️.