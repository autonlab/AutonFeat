# Multi-Feature Extraction

When featurizing time-series signals, it is often useful to extract multiple features from the same signal. For example, we may want to compute a subset of summary statistics (e.g. mean, variance, etc.) on the data. This often makes the job of the model easier in converging to a better performance metric. In this tutorial, we will show how to extract multiple features from the same signal using `AutonFeat`. We also show how the features from `AutonFeat` can be used in conjunction with other libraries such as `scikit-learn` to build a predictive model.

## Setup

First, we import the necessary packages and load the data.

```python
import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
```

Next, we load the Boston housing dataset and prepare it for featurization.

```python
boston = load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)
y = boston.target

# Convert the target to a binary variable
y = (y > np.median(y)).astype(int)
```

## Featurization

We now featurize the data using `AutonFeat`. We will extract the following features from the data:

- `mean`: The mean of the signal (at each window).
- `var`: The variance of the signal (at each window).
- `min`: The minimum value of the signal (at each window).
- `max`: The maximum value of the signal (at each window).
- `median`: The median of the signal (at each window).

### Sliding Window and Features

```python
import autonfeat as aft

# Sliding window parameters
window_size = 10
step_size = 1

# Create sliding window
window = aft.SlidingWindow(window_size=window_size, step_size=step_size)

# Define the transforms
transforms = [
    aft.MeanTransform(),
    aft.VarTransform(),
    aft.MinTransform(),
    aft.MaxTransform(),
    aft.MedianTransform(),
]
```

### Feature Extractor

```python
features = pd.DataFrame()

# Iterate over each transform
for transform in transforms:
    # Get the featurizer
    featurizer = window.use(transform)

    # Iterate over each column
    for col in X.columns:
        # Extract feature
        feature = featurizer(X[col])

        # Add feature to dataframe
        features[col + "_" + transform.name] = feature

# Add the computed features to the original data
X = pd.concat([X, features], axis=1)
```
Its that easy! We now have a dataframe with the original features and the extracted features.

## Model Training

We now train a model using the extracted features. We will use `scikit-learn` to train a logistic regression model.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    features, y, test_size=0.2, random_state=42
)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict_proba(X_test)[:, 1]
print("ROC AUC Score: {:.2f}".format(roc_auc_score(y_test, y_pred)))
```

## Conclusion

In this tutorial, we showed how to extract multiple features from the same signal using `AutonFeat`. We also showed how the features from `AutonFeat` can be used in conjunction with other libraries such as `scikit-learn` to build a predictive model.


If you enjoy using `AutonFeat`, please consider starring the [repository](https://github.com/autonlab/AutonFeat) ⭐️.

