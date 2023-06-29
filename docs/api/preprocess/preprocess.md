# Preprocessors

Data is sometimes not in the right form for a model. Hence, we featurize it. The same applies for a featurizer. Sometimes, data is not in the right form to be featurized. We may observe interesting properties emerge when we explore features in different domains. **This** is where the preprocess sub-module is helpful! We provide a wide variety of preprocessors to act as a bridge between the raw data and the featurizer.

## Preprocess Submodules

| Submodule | Description |
| --- | --- |
| `autofeat.preprocess` | Contains preprocessing featurizers in the form of modules (classes). |
| `autofeat.preprocess.functional` | Contains preprocessing featurizers in the form of functions. |

## Delta Distribution Shift

| Feature | Description | Endpoint |
| --- | --- | --- |
| Delta | Delta from a value and the rest of the signal | [`DeltaPreprocessor`](transform/delta_preprocessor.md) |
| Delta Mean | Delta from the mean of the signal | [`DeltaMeanPreprocessor`](transform/delta_mean_preprocessor.md) |
| Delta Median | Delta from the median of the signal | [`DeltaMedianPreprocessor`](transform/delta_median_preprocessor.md) |
| Delta Max | Delta from the maximum value of the signal | [`DeltaMaxPreprocessor`](transform/delta_max_preprocessor.md) |
| Delta Min | Delta from the minimum value of the signal | [`DeltaMinPreprocessor`](transform/delta_min_preprocessor.md) |
| Delta Std | Delta from the standard deviation of the signal | [`DeltaStdPreprocessor`](transform/delta_std_preprocessor.md) |
| Delta Var | Delta from the variance of the signal | [`DeltaVarPreprocessor`](transform/delta_var_preprocessor.md) |
| Delta Quantile | Delta from the quantile of the signal | [`DeltaQuantilePreprocessor`](transform/delta_quantile_preprocessor.md) |


## Frequency Domain

| Feature | Description | Endpoint |
| --- | --- | --- |
| DFT | 1D Discrete Fourier Transform of the signal | [`DFTPreprocessor`](transform/dft_preprocessor.md) |

## Functional Form

A functional form for each of the transforms above is also provided for convenience. Check out the `autofeat.preprocess.functional` sub-module for more details.

## Custom Preprocessors

`AutoFeat` makes it easy to design custom preprocessors by inheriting from the `Preprocess` class that is a part of the library's core engine. In this example, we show how to implement a `DeltaPreprocessor` that shifts a signal by some $\delta$ value.

```python
import numpy as np
from typing import Union, Callable
from autofeat.core import Preprocess

class DeltaPreprocessor(Preprocess):
    """
    Preprocess the signal by shifting the `signal` by some `delta` value.
    """
    def __init__(self, name: str = "Delta") -> None:
        super().__init__(name=name)

    def __call__(self, signal: np.ndarray, delta: Union[int, float, np.int_, np.float_], where: Callable[[Union[int, float, np.int_, np.float_]], Union[bool, np.bool_]] = lambda x: not np.isnan(x)) -> np.ndarray:
        where_fn = np.vectorize(where)
        # Compute mask and multiply by distribution shift along axis
        mask = where_fn(x)
        shift = mask * delta
        return x - shift
```

The preprocessor can then be applied to a signal $x$ as follows:

```python
# Define the signal
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

# Define the delta
delta = 1
delta_preprocessor = DeltaPreprocessor()

# Apply the preprocessor
processed_signal = delta_preprocessor(x, delta=delta)

# See the result
print(processed_signal)
```

```bash
[0 1 2 3 4 5 6 7 8]
```


See [this](../../tutorials/tutorials.md) for more examples on how to use preprocessors in `autofeat`.


If you enjoy using `AutoFeat`, please consider starring the [repository](https://github.com/autonlab/AutoFeat) ⭐️.