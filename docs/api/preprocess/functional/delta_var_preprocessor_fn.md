# Delta Var Preprocessor

The *delta var preprocessor* function shifts the input signal by the var of the signal. The is defined as:

$$
x_{shifted_{i}} = x_{i} - \sigma^{2}_{x}, \quad \forall i \in \{1, \dots, N\}
$$

For shifting signals by a custom $\delta$, see the [delta preprocessor](../functional/delta_preprocessor_fn.md) function. For more on how we compute the var of a signal, check out [var](../../functional/var.md) function.

::: autofeat.preprocess.functional.delta_var_tf

## Examples

Here we look at an example where we shift two signals by their var to demonstrate the effect of the *delta var preprocessor* function.

### Transform Signal

First, we define the signals as two normal distributions with different means and variances. Then, we apply the *delta var preprocessor* function to both signals.

A univariate normal distribution with mean $\mu$ and var $\sigma$ is defined as:

$$
\mathcal{N}(\mu, \sigma) = \frac{1}{\sigma \sqrt{2 \pi}} e^{-\frac{(x - \mu)^{2}}{2 \sigma^{2}}}
$$


```python
import numpy as np
import autofeat.preprocess.functional as PF

# Number of samples
n_samples = 100

# Generate sample data
x1 = np.random.normal(0, 1, n_samples)
x2 = np.random.normal(5, 5, n_samples)

shifted_x1 = PF.delta_var_tf(x1)
shifted_x2 = PF.delta_var_tf(x2)
```

### Visualize Transform

Next, we visualize the effect of the transform on the signals.

```python
import matplotlib.pyplot as plt

# Plot original data
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.plot(x1, '.', color='green', label='x1')
plt.plot(x2, '.', color='orange', label='x2')
plt.legend()
plt.title('Original Data')

# Plot shifted data
plt.subplot(1, 2, 2)
plt.plot(shifted_x1, '.', color='green', label='x1 shifted')
plt.plot(shifted_x2, '.', color='orange', label='x2 shifted')
plt.legend()
plt.title('Shifted Data')

plt.tight_layout()
plt.show()
```

This can be seen in the figure below.

![DeltaVar](../../../assets/delta_var_f_visualize.png)


If you enjoy using `AutoFeat`, please consider starring the [repository](https://github.com/autonlab/AutoFeat) ⭐️.