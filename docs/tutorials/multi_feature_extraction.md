# Multi Feature Extraction

In this tutorial we will show you how to install and use the `AutonFeat` package for multi-feature extraction. We apply the package to the problem of ***time-series forecasting*** i.e. using past values of a time-series to predict future values. When featurizing time-series signals, it is often useful to extract multiple features from the same signal. For example, we may want to compute a subset of summary statistics (e.g. mean, variance, etc.) on the data. This often makes the job of the model easier in converging to a better performance metric.

Feel free to follow along in this Google Colab notebook - 

<a href="https://colab.research.google.com/github/autonlab/AutonFeat/blob/main/examples/tutorials/multi_feature_extraction.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

## Install Packages


```python
%%capture
!pip install git+https://github.com/autonlab/AutonFeat.git
```


```python
%%capture
!git clone "https://github.com/autonlab/AutonFeat.git"
import sys
sys.path.append("AutonFeat")
```


```python
import autonfeat as aft
import numpy as np
import pandas as pd
```

## Load Dataset

List available datasets that we can use for this tutorial:


```python
print(aft.utils.datasets.list_datasets())
```

    ['airline passengers']


Load the ***Airline Passengers*** dataset:


```python
air_passengers_df = aft.utils.datasets.get_dataset(name='airline passengers')
air_passengers_df['datestamp'] = pd.to_datetime(air_passengers_df['datestamp'])
air_passengers_df.drop(columns=['uid'], inplace=True)
air_passengers_df.head()
```


    ---------------------------------------------------------------------------

    FileNotFoundError                         Traceback (most recent call last)

    Cell In[5], line 1
    ----> 1 air_passengers_df = aft.utils.datasets.get_dataset(name='airline passengers')
          2 air_passengers_df['datestamp'] = pd.to_datetime(air_passengers_df['datestamp'])
          3 air_passengers_df.drop(columns=['uid'], inplace=True)


    File ~/Work/CMU/AutonFeat/.venv_test/lib/python3.9/site-packages/autonfeat/utils/datasets/dataset.py:49, in get_dataset(name)
         47 dataset_map = get_dataset_map()
         48 dataset_path = dataset_map[name]
    ---> 49 return pd.read_csv(dataset_path)


    File ~/Work/CMU/AutonFeat/.venv_test/lib/python3.9/site-packages/pandas/io/parsers/readers.py:912, in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)
        899 kwds_defaults = _refine_defaults_read(
        900     dialect,
        901     delimiter,
       (...)
        908     dtype_backend=dtype_backend,
        909 )
        910 kwds.update(kwds_defaults)
    --> 912 return _read(filepath_or_buffer, kwds)


    File ~/Work/CMU/AutonFeat/.venv_test/lib/python3.9/site-packages/pandas/io/parsers/readers.py:577, in _read(filepath_or_buffer, kwds)
        574 _validate_names(kwds.get("names", None))
        576 # Create the parser.
    --> 577 parser = TextFileReader(filepath_or_buffer, **kwds)
        579 if chunksize or iterator:
        580     return parser


    File ~/Work/CMU/AutonFeat/.venv_test/lib/python3.9/site-packages/pandas/io/parsers/readers.py:1407, in TextFileReader.__init__(self, f, engine, **kwds)
       1404     self.options["has_index_names"] = kwds["has_index_names"]
       1406 self.handles: IOHandles | None = None
    -> 1407 self._engine = self._make_engine(f, self.engine)


    File ~/Work/CMU/AutonFeat/.venv_test/lib/python3.9/site-packages/pandas/io/parsers/readers.py:1661, in TextFileReader._make_engine(self, f, engine)
       1659     if "b" not in mode:
       1660         mode += "b"
    -> 1661 self.handles = get_handle(
       1662     f,
       1663     mode,
       1664     encoding=self.options.get("encoding", None),
       1665     compression=self.options.get("compression", None),
       1666     memory_map=self.options.get("memory_map", False),
       1667     is_text=is_text,
       1668     errors=self.options.get("encoding_errors", "strict"),
       1669     storage_options=self.options.get("storage_options", None),
       1670 )
       1671 assert self.handles is not None
       1672 f = self.handles.handle


    File ~/Work/CMU/AutonFeat/.venv_test/lib/python3.9/site-packages/pandas/io/common.py:859, in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
        854 elif isinstance(handle, str):
        855     # Check whether the filename is to be opened in binary mode.
        856     # Binary mode does not support 'encoding' and 'newline'.
        857     if ioargs.encoding and "b" not in ioargs.mode:
        858         # Encoding
    --> 859         handle = open(
        860             handle,
        861             ioargs.mode,
        862             encoding=ioargs.encoding,
        863             errors=errors,
        864             newline="",
        865         )
        866     else:
        867         # Binary mode
        868         handle = open(handle, ioargs.mode)


    FileNotFoundError: [Errno 2] No such file or directory: '/Users/dhruvsrikanth/Work/CMU/AutonFeat/.venv_test/lib/python3.9/site-packages/autonfeat/utils/datasets/data/airline_passengers.csv'


Plot the time-series (it is often useful to visualize the data before we start modeling):


```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1, figsize = (10, 5))
plot_df = air_passengers_df.set_index('datestamp')

plot_df[['passengers']].plot(ax=ax, linewidth=2)
ax.set_title('Airline Passengers Forecast', fontsize=22)
ax.set_ylabel('Monthly Passengers', fontsize=20)
ax.set_xlabel('Timestamp (t)', fontsize=20)
ax.legend(prop={'size': 15})
ax.grid()
```

## Preprocess Data

We can compute lag features using the preprocessing library from `AutonFeat`. This is often useful for time-series forecasting problems. We can combine this with feature extraction of summary statistics to create a rich set of features for our model (we will see this in the next section).


```python
# Define the number of lagged values to include as features
lags = [1, 2, 3]

preprocessor = aft.preprocess.LagPreprocessor()

# Create lagged features
for lag in lags:
    air_passengers_df[f'passengers_lag{lag}'] = preprocessor(air_passengers_df['passengers'].values, lag=lag)
```

## Feature Extraction

Set the parameters of the sliding window to perform feature extraction:

- `window_size`: the size of the sliding window
- `step_size`: the stride of the sliding window
- `overflow`: what to do with the last window if it is smaller than `window_size`

In this case, we choose to use the `stop` overflow strategy which means that we will stop the sliding window if the last window is smaller than `window_size`.


```python
# Sliding Window
window_size = 12
step_size = 1
sliding_window = aft.SlidingWindow(window_size=window_size, step_size=step_size, overflow='stop')
```

Define the features we want to extract from the time-series. We will extract the following features from the data:

- `mean`: The mean of the signal (at each window).
- `min`: The minimum value of the signal (at each window).
- `max`: The maximum value of the signal (at each window).
- `median`: The median of the signal (at each window).


```python
transforms = [
    aft.MeanTransform(),
    aft.MinTransform(),
    aft.MaxTransform(),
    aft.MedianTransform(),
]

feature_names = []
for transform in transforms:
    featurizer = sliding_window.use(transform=transform)
    features = featurizer(air_passengers_df['passengers'].values)

    # We add NaNs to the end of the array to make it the same length of the original array
    features = np.append(features, np.repeat(np.nan, len(air_passengers_df) - len(features)))

    # Add the features to the dataframe
    air_passengers_df[f'passengers_{transform.get_name().lower()}'] = features
    feature_names.append(f'passengers_{transform.get_name().lower()}')
```

Here is what our data looks like after *preprocessing* and *feature extraction*:


```python
air_passengers_df.head()
```

Lets see what the features looks like:


```python
# Plot the mean feature with the original data
fig, ax = plt.subplots(1, 1, figsize = (10, 5))
plot_df = air_passengers_df.set_index('datestamp')
plot_cols = ['passengers'] + feature_names
plot_df[plot_cols].plot(ax=ax, linewidth=2)
ax.set_title('Airline Passengers Forecast', fontsize=22)
ax.set_ylabel('Monthly Passengers', fontsize=20)
ax.set_xlabel('Timestamp (t)', fontsize=20)
ax.legend(prop={'size': 15})
ax.grid()
```

Feature extraction often results in a decrease in the number of data points. In our case, we chose to handle *overflow* using the `stop` strategy as specified in the `SlidingWindow`. We can see this reduction in the number of data points below:


```python
# Drop rows with missing values due to the lag
print(f'Dataframe contained {air_passengers_df.shape[0]} samples')
air_passengers_df.dropna(inplace=True)
print(f'Dataframe now contains {air_passengers_df.shape[0]} samples')
```

Finally, we split our data into training and test sets.


```python
from sklearn.model_selection import train_test_split

# Split the data into train and test sets
train_df, test_df = train_test_split(air_passengers_df, test_size=0.2, shuffle=False)

# Define the target variable
target = 'passengers'

# Define the features to use
features = ['passengers_lag1', 'passengers_lag2', 'passengers_lag3'] + feature_names

# Create the feature matrix and target vector
X_train = train_df[features].values
y_train = train_df[target].values.reshape(-1, 1)

X_test = test_df[features].values
y_test = test_df[target].values.reshape(-1, 1)

print('Training set sizes - ', X_train.shape, y_train.shape)
print('Test set sizes - ', X_test.shape, y_test.shape)
```

## Fit Model

We will be using an autoregressive model for time series forecasting. Autoregressive models are a class of models that use past values to predict future values, then use the predicted values to predict even further into the future. One of the simplest autoregressive models is a linear regression model. We will be using the `LinearRegression` class from `sklearn` to fit a linear regression model to our data.


```python
%%capture
from sklearn.linear_model import LinearRegression

# Fit a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)
```

## Evaluate Model

Remember that we adjusted the target values when we performed feature extraction? Well we can now use the remaining data points to evaluate our model. We will be using the ***mean absolute error*** (MAE) as our evaluation metric. The MAE is defined as:

$$
\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
$$

where $y_i$ is the actual value of the time-series at time $i$ and $\hat{y}_i$ is the predicted value of the time-series at time $i$.


```python
from sklearn.metrics import mean_absolute_error

# Predict on the test set
y_pred = model.predict(X_test)
```

We can evaluate the results both quantitatively and qualitatively. First, lets look at the MAE:


```python
# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae:.2f}')
```

If we compare this error to that of single-feature extraction (see the single feature extraction tutorial), we are able to get a better performance with multi-feature extraction as the $MAE = 32$ for single-feature extraction and the $MAE = 28$ for multi-feature extraction.

Next lets plot the actual values of the time-series against the predicted values:


```python
fig, ax = plt.subplots(1, 1, figsize = (10, 5))
plot_df = test_df.set_index('datestamp')
plot_df['passengers_pred'] = y_pred

plot_df[['passengers', 'passengers_pred']].plot(ax=ax, linewidth=2)
ax.set_title('Airline Passengers Forecast', fontsize=22)
ax.set_ylabel('Monthly Passengers', fontsize=20)
ax.set_xlabel('Timestamp (t)', fontsize=20)
ax.legend(prop={'size': 15})
ax.grid()

```

Here is what it looks like with respect to the original time-series:


```python
# Plot the training and test sets
fig, ax = plt.subplots(1, 1, figsize = (10, 5))
test_df['passengers_pred'] = y_pred
plot_df = pd.concat([train_df, test_df]).set_index('datestamp')

plot_df[['passengers', 'passengers_pred']].plot(ax=ax, linewidth=2)
ax.set_title('Airline Passengers Forecast', fontsize=22)
ax.set_ylabel('Monthly Passengers', fontsize=20)
ax.set_xlabel('Timestamp (t)', fontsize=20)
ax.legend(prop={'size': 15})
ax.grid()

```

If you enjoy using `AutonFeat`, please consider starring the [repository](https://github.com/autonlab/AutonFeat) ⭐️.
