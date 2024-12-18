# Feature Interaction Charting
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/autogluon/autogluon/blob/master/docs/tutorials/eda/eda-auto-analyze-interaction.ipynb)
[![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/autogluon/autogluon/blob/master/docs/tutorials/eda/eda-auto-analyze-interaction.ipynb)

This tool is made for quick interactions visualization between variables in a dataset. User can specify the variables to
be plotted on the x, y and hue (color) parameters. The tool automatically picks chart type to render based on the
detected variable types and renders 1/2/3-way interactions.

This feature can be useful in exploring patterns, trends, and outliers and potentially identify good predictors for the
task.

## Using Interaction Charts for Missing Values Filling

Let's load the titanic dataset:


```python
!pip install autogluon.eda

```


```python
import pandas as pd

df_train = pd.read_csv('https://autogluon.s3.amazonaws.com/datasets/titanic/train.csv')
df_test = pd.read_csv('https://autogluon.s3.amazonaws.com/datasets/titanic/test.csv')
target_col = 'Survived'
```

Next we will look at missing data in the variables:


```python
import autogluon.eda.auto as auto

auto.missing_values_analysis(train_data=df_train)
```

It looks like there are only two null values in the `Embarked` feature. Let's see what those two null values are:


```python
df_train[df_train.Embarked.isna()]
```

We may be able to fill these by looking at other independent variables. Both passengers paid a `Fare` of `$80`, are
of `Pclass` `1` and `female` `Sex`. Let's see how the `Fare` is distributed among all `Pclass` and `Embarked` feature
values:


```python
auto.analyze_interaction(train_data=df_train, x='Embarked', y='Fare', hue='Pclass')
```

The average `Fare` closest to `$80` are in the `C` `Embarked` values where `Pclass` is `1`. Let's fill in the missing
values as `C`.

## Using Interaction Charts To Learn Information About the Data


```python
state = auto.partial_dependence_plots(df_train, label='Survived', return_state=True)
```

A few observations can be made from the charts above:
- `Sex` feature has a very strong impact on the prediction result
- `Parch` has almost no impact on the outcome except when it is `0` or `1` - this is a candidate for clipping
- `Fare` and `Age`: both have a non-linear relationship with the outcome; `Fare` has two modes (density of blue lines) - these are good candidates to explore for feature interaction with other properties

Let's take a look at the two-way partial dependence plots to visualize any potential interactions between the two features. Here are some cases when it's a good idea to use two-way PDP:

* **Suspected interactions**: Even if two features are not highly correlated, they may still interact in the context of the model. If you suspect that there might be interactions between any two features, two-way PDP can help to verify the hypotheses.

* **Moderate to high correlation**: If two features have a moderate to high correlation, a two-way PDP can show how the combined effect of these features influences the model's predictions. In this case, the plot can help reveal whether the relationship between the features is additive, multiplicative, or more complex.

* **Complementary features**: If two features provide complementary information, a two-way PDP can help illustrate how the joint effect of these features impacts the model's predictions. For example, if one feature measures the length of an object and another measures its width, a two-way PDP could show how the area affects the predicted outcome.

* **Domain knowledge**: If domain knowledge suggests that the relationship between two features might be important for the model's output, a two-way PDP can help to explore and validate these hypotheses.

* **Feature importance**: If feature importance analysis ranks both features high in the leaderboard, it might be beneficial to examine their joint effect on the model's predictions.


```python
auto.partial_dependence_plots(df_train, label='Survived', features=['Fare', 'Age'], two_way=True)
```

We can see these two features interact in the bottom left quadrant of the chart, but have almost no effect on each other in other areas. Areas where `Age < 45` and `Fare < 60` can be explored further.

Let's take a look interactions between the features with the highest importance. To do this, we'll fit a quick model:


```python
state = auto.quick_fit(train_data=df_train, label='Survived', render_analysis=False, return_state=True)
```


```python
state.model_evaluation.importance
```


```python
auto.partial_dependence_plots(df_train, label='Survived', features=['Fare', 'SibSp'], two_way=True, show_help_text=False)
```

On this chart we see the features don't interact at all when `SibSp > 3` (Number of Siblings/Spouses Aboard), but they do have non-linear interaction for smaller groups. Those who were traveling in smaller groups had higher chances to escape. Those chances were even higher if the `Fare` paid was higher.

Let's analyze other variables higlighted above.


```python
auto.analyze_interaction(x='Parch', hue='Survived', train_data=df_train)
```


```python
auto.analyze_interaction(x='Pclass', y='Survived', train_data=df_train, test_data=df_test)
```

It looks like `63%` of first class passengers survived, while; `48%` of second class and only `24%` of third class 
passengers survived. Similar information is visible via `Fare` variable:

### `Fare` and `Age` features exploration

Because PDP plots hinted non-linear interaction in these two variables, let's take a closer look and visualize them individually and in jointly.


```python
auto.analyze_interaction(x='Fare', hue='Survived', train_data=df_train, test_data=df_test, chart_args=dict(fill=True))
```


```python
auto.analyze_interaction(x='Age', hue='Survived', train_data=df_train, test_data=df_test)
```

The very left part of the distribution on this chart possibly hints that children and infants were the priority.


```python
auto.analyze_interaction(x='Fare', y='Age', hue='Survived', train_data=df_train, test_data=df_test)
```

This chart highlights three outliers with a Fare of over `$500`. Let's take a look at these:


```python
df_train[df_train.Fare > 400]
```

As you can see all 4 passengers share the same ticket. Per-person fare would be 1/4 of this value. Looks like we can 
add a new feature to the dataset fare per person; also this allows us to see if some passengers travelled in larger 
groups. Let's create two new features and take at the Fare-Age relationship once again.


```python
ticket_to_count = df_train.groupby(by='Ticket')['Embarked'].count().to_dict()
data = df_train.copy()
data['GroupSize'] = data.Ticket.map(ticket_to_count)
data['FarePerPerson'] = data.Fare / data.GroupSize

auto.analyze_interaction(x='FarePerPerson', y='Age', hue='Survived', train_data=data)
auto.analyze_interaction(x='FarePerPerson', y='Age', hue='Pclass', train_data=data)
```

You can see cleaner separation between `Fare`, `Pclass` and `Survived` now.
