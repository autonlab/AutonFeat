[![Typing SVG](https://readme-typing-svg.demolab.com?font=Georgia&weight=900&size=26&duration=4000&pause=1000&color=F71313&center=false&vCenter=true&width=435&lines=AutoFeat)](https://git.io/typing-svg)


# AutoFeat

A domain agnostic package for time series featurization.

We've included some domain specific featurizers for popular domains like healthcare. 

In addition, we've included some preprocessors to help with featurization.

Finally, our goal was to make this package without too many dependencies and overhead.


## Design Objectives

- **Simple**: The package should be easy to use and require as little user input as possible.
- **Interpretable**: The package should be interpretable, i.e. the user should be able to understand the featurization process and the resulting features through good documentation.
- **Fast**: The package should be fast enough to be used in production.
- **Flexible**: The package should be flexible enough to be used in a variety of settings, including custom featurization functions.


## Assumptions

- The input data is a 1D time series in the form of a numpy array.
- If there are missing values, they must be represented by `np.nan` to be detected, otherwise, gaps in the time series are not detected.

## Installation

*(Coming soon)*
```bash
pip install autofeat
```


## Domain Agnostic Features


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
| Skewness | Skewness of the signal | `SkewnessTransform` |
| Kurtosis | Kurtosis of the signal | `KurtosisTransform` |


**Data Sparsity Measures**:
| Feature | Description | Endpoint |
| --- | --- | --- |
| Data Density | Ratio of valid values to window size | `DataDensityTransform` |
| Data Sparsity | Ratio of missing values to window size | `DataSparsityTransform` |


**Information Theoretic Measures**:
| Feature | Description | Endpoint |
| --- | --- | --- |
| Shannon Entropy | Shannon entropy of the signal | `EntropyTransform` |
| KL Divergence | KL divergence (relative entropy) of the signal with respect to another distribution. | `EntropyTransform` |
| Sample Entropy | Sample entropy of the signal | `SampleEntropyTransform` |
| Cross Entropy | Cross entropy of the signal with respect to another distribution | `CrossEntropyTransform` |


### Domain Specific

**Biomedical, and Physiological Signals**:
| Feature | Description | Endpoint |
| --- | --- | --- |


#### Example

```python
import numpy as np
import sys
import autofeat as aft

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


**Frequency Domain**:
| Feature | Description | Endpoint |
| --- | --- | --- |
| DFT | 1D Discrete Fourier Transform of the signal | `DFTPreprocessor` |


#### Example

```python
import numpy as np
import sys
import autofeat as aft

def main():
    # Random distribution
    n_samples = 100
    dist = np.random.rand(n_samples)

    # Preprocessor
    preprocessor = aft.preprocess.DeltaMaxPreprocessor()

    # Transform distribution
    transformed_dist = preprocessor(dist)

    # Print features
    print(preprocessor)
    print(dist)
    print(transformed_dist)

if __name__ == '__main__':
    main()
```

## Testing

All tests can be run using the following command:

```bash
python -m pytest autofeat_test
```

## Contributing

When contributing, please add tests to the `autofeat_test` directory. Additionally, I follow `flake8` as a linter. Please lint your code before submitting a pull request to maintain design consistency.

The following commands can be run for verficiation before opening a PR:

```bash
# Unit tests
python -m pytest autofeat_test

# Linting
flake8 autofeat --ignore=E501
```


