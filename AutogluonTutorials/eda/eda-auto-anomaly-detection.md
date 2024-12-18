# Anomaly Detection Analysis
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/autogluon/autogluon/blob/master/docs/tutorials/eda/eda-auto-anomaly-detection.ipynb)
[![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/autogluon/autogluon/blob/master/docs/tutorials/eda/eda-auto-anomaly-detection.ipynb)

Anomaly detection is a powerful technique used in data analysis and machine learning to identify unusual patterns or behaviors that deviate from the norm. These deviations, known as anomalies or outliers, can be indicative of errors, fraud, system failures, or other exceptional events. By detecting these anomalies early, organizations can take proactive measures to address potential issues, enhance security, optimize processes, and make more informed decisions. In this tutorial, we will introduce anomaly detection tools available in AutoGluon EDA package and showcase how to identify these irregularities within your data, even if you're new to the subject.


```python
!pip install autogluon.eda
!pip install autogluon.tabular[lightgbm]
```


```python
import pandas as pd
import seaborn as sns

import autogluon.eda.auto as auto
```

### Loading and pre-processing the data

First we will load the data. We will use the Titanic dataset.


```python
df_train = pd.read_csv('https://autogluon.s3.amazonaws.com/datasets/titanic/train.csv')
df_test = pd.read_csv('https://autogluon.s3.amazonaws.com/datasets/titanic/test.csv')
target_col = 'Survived'
```

`auto.detect_anomalies` will automatically preprocess the data, but it doesn't fill in missing numeric values by default. We'll take care of filling those in ourselves before feeding the data into the anomaly detector.


```python
x = df_train
x_test = df_test
x.Age.fillna(x.Age.mean(), inplace=True)
x_test.Age.fillna(x.Age.mean(), inplace=True)
x_test.Fare.fillna(x.Fare.mean(), inplace=True)
```

### Running Initial Anomaly Analysis


```python
# This parameter specifies how many standard deviations above mean anomaly score are considered
# to be anomalies (only needed for visualization, does not affect scores calculation).
threshold_stds = 3
```


```python
auto.detect_anomalies(
    train_data=x,
    test_data=x_test,
    label=target_col,
    threshold_stds=threshold_stds,
    show_top_n_anomalies=None,
    fig_args={
        'figsize': (6, 4)
    },
    chart_args={
        'normal.color': 'lightgrey',
        'anomaly.color': 'orange',
    }
)
```

### Handling Covariate Shift
The test data chart appears to show increasing anomaly scores as we move through the records. This is not normal; let's check for a covariate shift.


```python
auto.covariate_shift_detection(train_data=x, test_data=x_test, label=target_col)
```


```python
ax = sns.lineplot(data=df_train[['PassengerId']].reset_index(), x='index', y='PassengerId', label='Train')
sns.lineplot(ax=ax, data=df_test[['PassengerId']].reset_index(), x='index', y='PassengerId', label='Test');
```

This feature looks like a monotonically increasing ID and carries no value for our problem; we are going to remove it.


```python
x = x.drop(columns=['PassengerId'], errors='ignore')
x_test = x_test.drop(columns=['PassengerId'], errors='ignore')
```


```python
auto.covariate_shift_detection(train_data=x, test_data=x_test, label=target_col)
```

### Run Anomaly Analysis on Cleaned Data


```python
state = auto.detect_anomalies(
    train_data=x,
    test_data=x_test,
    label=target_col,
    threshold_stds=3,
    show_top_n_anomalies=5,
    explain_top_n_anomalies=1,
    return_state=True,
    show_help_text=False,
    fig_args={
        'figsize': (6, 4)
    },
    chart_args={
        'normal.color': 'lightgrey',
        'anomaly.color': 'orange',
    }    
)
```

### Visualize Anomalies

As we can see from the feature impact charts, the anomaly scores are primarily influenced by the Fare and Age features. Let's take a look at a visual slice of the feature space. We can get the scores from state under `anomaly_detection.scores.<dataset>` keys:


```python
train_anomaly_scores = state.anomaly_detection.scores.train_data
test_anomaly_scores = state.anomaly_detection.scores.test_data
```


```python
auto.analyze_interaction(train_data=df_train.join(train_anomaly_scores), x="Fare", y="Age", hue="score", chart_args=dict(palette='viridis'))
```


```python
auto.analyze_interaction(train_data=df_test.join(test_anomaly_scores), x="Fare", y="Age", hue="score", chart_args=dict(palette='viridis'))
```

The data points in the lower left corner don't appear to be anomalies. However, this is only because we are looking at a slice of the 11-dimensional data. While it might not seem like an anomaly in this slice, it is salient in other dimensions.

In conclusion, in this tutorial we've guided you through the process of using AutoGluon for anomaly detection. We've covered how to automatically detect anomalies with just a few lines of code. We also explored finding and visualizing the top detected anomalies, which can help you better understand and address the underlying issues. Lastly, we explored how to find the main contributing factors that led to a data point being marked as an anomaly, allowing you to pinpoint the root causes and take appropriate action.
