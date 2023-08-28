<!-- 
Author(s): Dhruv Srikanth
Email(s): dsrikant (at) andrew (dot) cmu (dot) edu
Acknowledgements:
Copyright (c) 2023 Carnegie Mellon University, Auton Lab
This code is subject to the license terms contained in the code repo.
-->

# Transform

One of the ***core*** building blocks of [`AutonFeat`](../../index.md) is the [`Transform`](transform.md) abstraction. This enables users to define custom featurizers that can be applied to the sliding window intervals and is how we implement the build-in feature extractors. 

::: autonfeat.core.Transform

## Examples

In the below example, we show how to define a custom featurizer that computes the mean of the signal.

### Define Featurizer Function

For ease of use, we define a function that computes the mean of the signal.

```python
import numpy as np

def mean_function(x):
    return np.mean(x)
```

### Define Transform

Use the `Transform` abstraction to define the featurizer.

```python
import numpy as np
from typing import Callable, Union
from autonfeat.core import Transform

class MeanTransform(Transform):
    def __init__(self, name: str = "Mean") -> None:
        super().__init__(name=name)

    def __call__(self, signal_window: np.ndarray, where: Callable[[Union[int, float, np.int_, np.float_]], Union[bool, np.bool_]] = lambda x: not np.isnan(x)) -> Union[np.float_, np.int_]:
        where_fn = np.vectorize(pyfunc=where)
        filtered_signal_window = signal_window[where_fn(signal_window)]
        return mean_function(filtered_signal_window)
```

### Apply Transform

Using the [`SlidingWindow`](fixed_window.md) abstraction, we can apply the transform to the sliding window intervals.

```python
import autonfeat as aft

# Random data
n_samples = 100
x = np.random.rand(n_samples)

# Create sliding window
ws = 10
ss = 10
window = aft.SlidingWindow(window_size=ws, step_size=ss)

# Create transform
tf = MeanTransform()

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
