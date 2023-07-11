# Single Feature Extraction

In this tutorial we will show you how to install and use the `AutonFeat` package for single-feature extraction. We apply the package to the problem of ***time-series forecasting*** i.e. using past values of a time-series to predict future values.

Feel free to follow along in this Google Colab notebook - 

<a href="https://colab.research.google.com/github/autonlab/AutonFeat/blob/main/examples/tutorials/single_feature_extraction.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

## Install Packages


```python
%%capture
!pip install autonfeat
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




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>datestamp</th>
      <th>passengers</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1949-01-31</td>
      <td>112.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1949-02-28</td>
      <td>118.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1949-03-31</td>
      <td>132.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1949-04-30</td>
      <td>129.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1949-05-31</td>
      <td>121.0</td>
    </tr>
  </tbody>
</table>
</div>



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


    
![png](single_feature_extraction_files/single_feature_extraction_12_0.png)
    


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

Define the feature we want to extract from the time-series. Here we will be considering the ***mean*** of the time-series values in each window. For a particular window of size $w$, the mean is defined as:

$$
\text{mean} = \frac{1}{w} \sum_{i=1}^{w} x_i
$$

where $x_i$ is the value of the time-series at time $i$.


```python
feature_extractor = aft.MeanTransform()
featurizer = sliding_window.use(feature_extractor)
features = featurizer(air_passengers_df['passengers'].values)

# We add NaNs to the end of the array to make it the same length of the original array
features = np.append(features, np.repeat(np.nan, len(air_passengers_df) - len(features)))

# Add the features to the dataframe
air_passengers_df['passengers_mean'] = features
```

Here is what our data looks like after *preprocessing* and *feature extraction*:


```python
air_passengers_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>datestamp</th>
      <th>passengers</th>
      <th>passengers_lag1</th>
      <th>passengers_lag2</th>
      <th>passengers_lag3</th>
      <th>passengers_mean</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1949-01-31</td>
      <td>112.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>126.666667</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1949-02-28</td>
      <td>118.0</td>
      <td>112.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>126.916667</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1949-03-31</td>
      <td>132.0</td>
      <td>118.0</td>
      <td>112.0</td>
      <td>NaN</td>
      <td>127.583333</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1949-04-30</td>
      <td>129.0</td>
      <td>132.0</td>
      <td>118.0</td>
      <td>112.0</td>
      <td>128.333333</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1949-05-31</td>
      <td>121.0</td>
      <td>129.0</td>
      <td>132.0</td>
      <td>118.0</td>
      <td>128.833333</td>
    </tr>
  </tbody>
</table>
</div>



Lets see what the ***mean*** feature looks like:


```python
# Plot the mean feature with the original data
fig, ax = plt.subplots(1, 1, figsize = (10, 5))
plot_df = air_passengers_df.set_index('datestamp')

plot_df[['passengers', 'passengers_mean']].plot(ax=ax, linewidth=2)
ax.set_title('Airline Passengers Forecast', fontsize=22)
ax.set_ylabel('Monthly Passengers', fontsize=20)
ax.set_xlabel('Timestamp (t)', fontsize=20)
ax.legend(prop={'size': 15})
ax.grid()
```


    
![png](single_feature_extraction_files/single_feature_extraction_24_0.png)
    


Feature extraction often results in a decrease in the number of data points. In our case, we chose to handle *overflow* using the `stop` strategy as specified in the `SlidingWindow`. We can see this reduction in the number of data points below:


```python
# Drop rows with missing values due to the lag
print(f'Dataframe contained {air_passengers_df.shape[0]} samples')
air_passengers_df.dropna(inplace=True)
print(f'Dataframe now contains {air_passengers_df.shape[0]} samples')
```

    Dataframe contained 144 samples
    Dataframe now contains 130 samples


Finally, we split our data into training and test sets.


```python
from sklearn.model_selection import train_test_split

# Split the data into train and test sets
train_df, test_df = train_test_split(air_passengers_df, test_size=0.2, shuffle=False)

# Define the target variable
target = 'passengers'

# Define the features to use
features = ['passengers_lag1', 'passengers_lag2', 'passengers_lag3', 'passengers_mean']

# Create the feature matrix and target vector
X_train = train_df[features].values
y_train = train_df[target].values.reshape(-1, 1)

X_test = test_df[features].values
y_test = test_df[target].values.reshape(-1, 1)

print('Training set sizes - ', X_train.shape, y_train.shape)
print('Test set sizes - ', X_test.shape, y_test.shape)
```

    Training set sizes -  (104, 4) (104, 1)
    Test set sizes -  (26, 4) (26, 1)


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

    Mean Absolute Error: 32.31


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


    
![png](single_feature_extraction_files/single_feature_extraction_38_0.png)
    


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


    
![png](single_feature_extraction_files/single_feature_extraction_40_0.png)
    


If you enjoy using `AutonFeat`, please consider starring the [repository](https://github.com/autonlab/AutonFeat) ⭐️.
