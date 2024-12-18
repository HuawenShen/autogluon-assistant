# Automated Target Variable Analysis
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/autogluon/autogluon/blob/master/docs/tutorials/eda/eda-auto-target-analysis.ipynb)
[![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/autogluon/autogluon/blob/master/docs/tutorials/eda/eda-auto-target-analysis.ipynb)

In this section we explore automated dataset overview functionality.

Automated target variable analysis aims to automatically analyze and summarize the variable we are trying to predict
(label). The goal of this analysis is to take a deeper look into target variable structure and its relationship with
other important variables in the dataset.

To simplify discovery of outliers and useful patterns, this functionality introduces components which allow generating
descriptive statistics and visualizing the target distribution and relationships between the target variable and other
variables in the dataset.

## Classification Example

We will start with getting the titanic dataset and performing a quick one-line overview to get the information.


```python
!pip install autogluon.eda

```


```python
import pandas as pd

df_train = pd.read_csv('https://autogluon.s3.amazonaws.com/datasets/titanic/train.csv')
df_test = pd.read_csv('https://autogluon.s3.amazonaws.com/datasets/titanic/test.csv')
target_col = 'Survived'
```

The report consists of multiple parts: statistical information overview enriched with feature types detection and
missing value counts focused only on the target variable.

Label Insights will highlight dataset features which require attention (i.e. class imbalance or out-of-domain data in
test dataset).

The next component is feature distribution visualization. This is helpful for choosing data transformations and/or
model selection. For regression tasks, the framework automatically fits multiple distributions available in scipy. The
distributions with the best fit will be displayed on the chart. Distributions information will be displayed below the
chart.

Next, the report will provide correlation analysis focusing only on highly-correlated features and visualization of
their relationships with the target.

To perform the analysis, we need just one line:


```python
import autogluon.eda.auto as auto

auto.target_analysis(train_data=df_train, label=target_col)
```

## Regression Example

In the previous section we tried a classification example. Let's try a regression. It has a few differences.


```python
df_train = pd.read_csv('https://autogluon.s3.amazonaws.com/datasets/AmesHousingPriceRegression/train_data.csv')
df_test = pd.read_csv('https://autogluon.s3.amazonaws.com/datasets/AmesHousingPriceRegression/test_data.csv')
target_col = 'SalePrice'

auto.target_analysis(
    train_data=df_train, label=target_col, 
    # Optional; default will try to fit all available distributions
    fit_distributions=['laplace_asymmetric', 'johnsonsu', 'exponnorm']  
)
```
