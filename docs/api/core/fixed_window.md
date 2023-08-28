<!-- 
Author(s): Dhruv Srikanth
Email(s): dsrikant (at) andrew (dot) cmu (dot) edu
Acknowledgements:
Copyright (c) 2023 Carnegie Mellon University, Auton Lab
This code is subject to the license terms contained in the code repo.
-->

# Fixed Sliding Window

A fixed sliding window is an abstraction that can be used to compute features across a signal using a sliding window. The window size and stride are fixed upon instantiation by the user. This operation can be represented as:

$$
f_{i} = \text{F}(x_{i - w}, \dots, x_{i - 1}, x_{i}, \dots, x_{i + w}), \quad \forall i \in \{w, \dots, N - w\}
$$

where $F$ represents a feature extractor function, $f_{i}$ represents the $i$th feature, $x_{i}$ represents the $i$th element of the input signal, $w$ represents the window size, and $N$ represents the number of elements in the signal.

Here's a visual illustration of the above:

![FixedSlidingWindow](../../assets/fixed_sliding_window_animation.gif)


Overflow can occur when the window extends beyond the bounds of the signal. This can be handled in a number of ways. The default behavior is to pad the signal with zeros. However, we provide several other options for handling overflow.

We provide some examples below on how to combine the *fixed sliding window* abstraction with feature extractors to compute features across a signal.

::: autonfeat.core.SlidingWindow

## Examples

### Problem Setup

Consider the following example with signal $x$, window $W$ and feature extractor $F$:

$t = \left[0, \dots, 10\right]$

$x = \sin(2t) + \cos(3t) + \sin(5t) + \cos(7t) + \exp(-t / 5)$

$W_{size} = 50$

$W_{stride} = 25$

$F = \text{mean}$

```python
import numpy as np
import autonfeat as aft

# Setup the signal
t = np.linspace(0, 10, 1000)
signal = np.sin(2 * t) + np.cos(3 * t) + np.sin(5 * t) + np.cos(7 * t) + np.exp(-t / 5)
```

### Setup Sliding Window and Feature Extractor

```python
# Setup the sliding window
window_size = 50
step_size = 25
sliding_window = aft.SlidingWindow(window_size=window_size, step_size=step_size)

# Setup the feature extractor
feature_extractor = aft.MeanTransform()

# Get the featurizer object
featurizer = sliding_window.use(feature_extractor)
```

### Extract Features

```python
# Extract features
features = featurizer(signal)

print(features)
```

We can view the following operation below:

![FixedSlidingWindow](../../assets/fixed_sliding_window_animation.gif)


If you enjoy using [`AutonFeat`](../../index.md), please consider starring the [repository](https://github.com/autonlab/AutonFeat) ⭐️.