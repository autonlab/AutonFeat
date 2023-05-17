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

def main():
    # Random data
    n_samples = 100
    x = np.random.rand(n_samples)
    
    # Create sliding window
    ws = 10
    ss = 10
    window = aft.SlidingWindow(window_size=ws, step_size=ss)

    # Create transform
    tf = aft.MeanTransform()

    # Get featurizer
    featurizer = window.use(tf)

    # Get features
    features = featurizer(x)

    # Print features
    print(window)
    print(tf)
    print(features)

if __name__ == '__main__':
    main()

```

## Features

**Summary Statistics**: 
| Feature | Description | Endpoint |
| --- | --- | --- |
| Max | Maximum value of the signal | `MaxTransform` |
| Min | Minimum value of the signal | `MinTransform` |
| Mean | Mean of the signal | `MeanTransform` |
| Median | Median of the signal | `MedianTransform` |
| Standard Deviation | Standard deviation of the signal | `StdTransform` |
| Variance | Variance of the signal | `VarTransform` |
| Quantile | Quantile of the signal | `QuantileTransform` |
| Range | Range of the signal | `RangeTransform` |
| IQR | Interquartile range of the signal | `IQRTransform` |
| N Valid | Number of valid values in the signal | `NValidTransform` |

**Data Sparsity**:
| Feature | Description | Endpoint |
| --- | --- | --- |
| Data Density | Ratio of valid values to window size | `DataDensityTransform` |
| Data Sparsity | Ratio of missing values to window size | `DataSparsityTransform` |

## Preprocessors

**Delta Distribution Shift**:
| Feature | Description | Endpoint |
| --- | --- | --- |
| Delta | Delta from a value and the rest of the signal | `DeltaPreprocessor` |
| Delta Mean | Delta from the mean of the signal | `DeltameanPreprocessor` |
| Delta Median | Delta from the median of the signal | `DeltaMedianPreprocessor` |
| Delta Max | Delta from the maximum value of the signal | `DeltaMaxPreprocessor` |
| Delta Min | Delta from the minimum value of the signal | `DeltaMinPreprocessor` |
| Delta Std | Delta from the standard deviation of the signal | `DeltaStdPreprocessor` |
| Delta Var | Delta from the variance of the signal | `DeltaVarPreprocessor` |
| Delta Quantile | Delta from the quantile of the signal | `DeltaQuantilePreprocessor` |


