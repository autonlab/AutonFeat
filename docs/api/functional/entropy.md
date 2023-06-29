# Entropy Function

The entropy function computes the entropy of a distribution. The entropy is a measure of the uncertainty of a random variable. The entropy of a distribution is defined as:

$$
H(X) = -\sum_{i=1}^{n} p(x_i) \log p(x_i)
$$

where $p(x_i)$ is the probability of the $i$-th outcome. The entropy is maximized when all outcomes are equally likely. The entropy is zero when the distribution is deterministic.

We can use the entropy function to compute the entropy of a single discrete probability distribution using *Shannon Entropy*. We can also use the entropy function to compute the relative entropy between two discrete probability distributions. This is also called the *Kullback-Leibler (KL) divergence*. This is defined as:

$$
D_{KL}(p||q) = H(p, q) = \sum_{x} p(x) \log \frac{p(x)}{q(x)}
$$

where $p$ and $q$ are the two probability distributions. The relative entropy is zero if and only if the two distributions are identical. The relative entropy is always non-negative.

::: autofeat.functional.entropy_tf
      

## Examples

### Shannon Entropy

```python
import numpy as np
import autofeat as aft
import autofeat.functional as F

# Random data
n_samples = 100
x = np.random.randint(0, 10, n_samples)

# Sliding window
ws = 10
ss = 10
window = aft.SlidingWindow(window_size=ws, step_size=ss)

# Get featurizer
featurizer = window.use(F.entropy_tf)

# Get features
features = featurizer(x)

# Print features
print(features)
```

### KL Divergence

```python
import numpy as np
import autofeat as aft
import autofeat.functional as F

# Random data
n_samples = 100
x1 = np.random.rand(n_samples)
x2 = np.random.rand(n_samples)

# Sliding window
ws = 10
ss = 10
window = aft.SlidingWindow(window_size=ws, step_size=ss)

# Get featurizer
featurizer = window.use(F.entropy_tf)

# Get features
features = featurizer(x1, x2)

# Print features
print(features)
```

If you enjoy using `AutoFeat`, please consider starring the [repository](https://github.com/autonlab/AutoFeat) ⭐️.