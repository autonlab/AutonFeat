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

# Preprocess

The [`Preprocess`](preprocess.md) class is a ***core*** building block in [`AutonFeat`](../../index.md). This enables users to define custom preprocessors that can be applied to the signal before extracting features.

::: autonfeat.core.Preprocess

## Examples

In this example, we define a custom preprocessor that shifts the signal by some `delta` value.

### Define Custom Preprocessor

We define a custom preprocessor that performs this computation in the following way.

```python
import numpy as np
from typing import Union, Callable
from autonfeat.core import Preprocess

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

### Apply Custom Preprocessor

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

If you enjoy using [`AutonFeat`](../../index.md), please consider starring the [repository](https://github.com/autonlab/AutonFeat) ⭐️.