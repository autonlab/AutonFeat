# Delta Quantile Preprocessor Transform

The *Delta Quantile Preprocessor Transform* shifts the input signal by the quantile of the signal. This is defined as:

$$
x_{shifted_{i}} = x_{i} - \text{quantile}(x), \quad \forall i \in \{1, \dots, N\}
$$

where $x_{i}$ represents an element of the input signal, $x_{shifted_{i}}$ represents an element of the output signal, and $N$ is the number of elements in the signal.

For shifting signals by a custom $\delta$, see the [Delta Transform Preprocessor](delta_preprocessor.md). For more on how we compute the quantile of a signal, check out [quantile](../../functional/quantile.md) function.

::: autofeat.preprocess.transform.DeltaQuantilePreprocessor

## Examples

### Transform Signal

```python
import numpy as np
import autofeat as aft

# Generate data
n_samples = 1000
x = np.random.normal(-5, 5, n_samples)

# Create a preprocessor
preprocessor = aft.preprocess.DeltaQuantilePreprocessor()

q_tile = 0.25 # 25th quantile
# Preprocess data
x_shifted = preprocessor(x, q=q_tile)
```

### Visualize Transform

```python
import matplotlib.pyplot as plt

# Plot normal and shifted data
original_quantile = aft.functional.quantile_tf(x, q_tile)
shifted_quantile = aft.functional.quantile_tf(x_shifted, q_tile)

plt.figure(figsize=(8, 6))

plt.plot(x, '.', color='blue', label='Origianl Data')
plt.axhline(original_quantile, color='red', linestyle='--', linewidth=3, label=f'Original Data 25th quantile = {original_quantile:.2f}')

plt.plot(x_shifted, '.', color='orange', label='Shifted Data')
plt.axhline(shifted_quantile, color='green', linestyle='--', linewidth=3, label=f'Shifted Data 25th quantile = {shifted_quantile:.2f}')

plt.legend()
plt.title('Delta Quantile Preprocessing Transform')

plt.tight_layout()
plt.show()
```

![DeltaQuantile](../../../assets/delta_quantile_visualize.png)


If you enjoy using `AutoFeat`, please consider starring the [repository](https://github.com/autonlab/AutoFeat) ⭐️.