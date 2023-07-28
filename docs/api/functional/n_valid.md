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

# N-Valid Function

Compute the number of valid measurements in a sliding window. A valid measurement by default is defined as a measurement that is not `np.nan`, however this can be altered by passing a *validity* function to the argument `where`. The *validity* function should take a single argument, the measurement, and return `True` if the measurement is valid, and `False` otherwise. The function can be defined as:

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

::: autonfeat.functional.n_valid_tf
      

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
featurizer = window.use(F.n_valid_tf)

# Get features
features = featurizer(x)

# Print features
print(features)
```


If you enjoy using [`AutonFeat`](../../index.md), please consider starring the [repository](https://github.com/autonlab/AutonFeat) ⭐️.