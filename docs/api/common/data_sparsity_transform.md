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

# Data Sparsity Transform

The data sparsity transform computes the ratio of invalid values in a sliding window to the total number of values in the window. See [`NValidTransform`](n_valid_transform.md) for more details on how valid values are computed. Invalid values are computed by computing $1 - N_{valid}$, where $N_{valid}$ is the number of valid values in the signal window. It can be coupled with the [`SlidingWindow`](../core/fixed_window.md) abstraction to compute the `data_sparsity` feature of a time series. It can be defined as:

$$
\text{sparsity} = \frac{N_{invalid}}{N_{total}}
$$

where $N_{invalid}$ is the number of invalid values in a window $W$ and $N_{total}$ is the total number of values in $W$.


::: autonfeat.common.DataSparsityTransform
      

## Examples

```python
import numpy as np
import autonfeat as aft

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

If you enjoy using [`AutonFeat`](../../index.md), please consider starring the [repository](https://github.com/autonlab/AutonFeat) ⭐️.