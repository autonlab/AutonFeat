# Sample Entropy Function

The sample entropy function computes the sample entropy of a window. When combined with the [SlidingWindow](../core/fixed_window.md) abstraction, the mean function can be used to compute the `sample entropy` feature of a time series. Sample entropy is a measure of the complexity of the signal. It is a modification of the approximate entropy (ApEn) algorithm. It is defined as:

$$
\text{Sample Entropy} = -\log\left(\frac{A}{B}\right)
$$

where $A$ is the number of matches for template vectors of length $m$ and $B$ is the number of matches for template vectors of length $m + 1$. A match is defined as a template vector $x_{m_i}$ that is close to another template vector $x_{m_j}$ in the sense that the maximum absolute difference between their corresponding scalar elements is less than or equal to a threshold $r$.

::: autofeat.functional.sample_entropy_tf
      

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
featurizer = window.use(F.sample_entropy_tf)

# Get features
features = featurizer(x)

# Print features
print(features)
```


If you enjoy using `AutoFeat`, please consider starring the [repository](https://github.com/autonlab/AutoFeat) ⭐️.