# AutoFeat

A package for time series featurization, build with the following principles in mind:

1. **Simple**: The package should be easy to use and require as little user input as possible.
2. **Interpretable**: The package should be interpretable, i.e. the user should be able to understand the featurization process and the resulting features through good documentation.
3. **Fast**: The package should be fast enough to be used in production.
4. **Flexible**: The package should be flexible enough to be used in a variety of settings, including custom featurization functions.

We have tried to make it domain agnostic allowing for minimal dependencies.

Assumptions:
- The input data is a 1D time series in the form of a numpy array.
- If there are missing values, they must be represented by `np.nan` to be detected, otherwise, gaps in the time series are not detected.

## Example

```python
import autofeat as aft
import numpy as np

if __name__ == '__main__':
    # Random data
    n_samples = 100
    x = np.random.rand(n_samples)
    
    # Create sliding window
    ws = 10
    ss = 10
    window = aft.SlidingWindow(window_size=ws, step_size=ss)

    # Create transform
    mean_tf = aft.MaxTransform()

    # Get featurizer
    featurizer = window.use(mean_tf)

    # Get features
    features = featurizer(x)

    # Print features
    print(window)
    print(mean_tf)
    print(features)


```