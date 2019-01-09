
## Feature engineering always wins. Exp in XOR problem, create another feature as z= x*y, that's it.

## In time series data, fit a radial basis function for each month, and tune the `alpha`, which control the width of gaussian. So for one feature, we get 12 feature, but very-very helpful.

## Weighted Linear Regression: Give more importance to the current Year or to some season, when sales are on peak like gift-shoppin while valentine days or sth like that... (Put an exponential decaying as importance for each instances)


## Cyclic Features
Features which are repeated, like month, date, times and so on.
```python
import numpy as np

df['hr_sin'] = np.sin(df.hr*(2.*np.pi/24))
df['hr_cos'] = np.cos(df.hr*(2.*np.pi/24))
df['mnth_sin'] = np.sin((df.mnth-1)*(2.*np.pi/12))
df['mnth_cos'] = np.cos((df.mnth-1)*(2.*np.pi/12))
```

**NOTE**: Pasty is python package(lib), that can help in feature generation

## For categorical feature, can use **contrast Coding** 

```python
import category_encoders as ce
encoder = ce.BackwardDifferenceEncoder(cols=['cat_variable'])
data['cat_variable_encoding'] = encoder.fit_transform(data['cat_variable'])
```


## Frequency Encoding

- Using frequency or response rate: Combining levels based on business logic is effective but we may always not have the domain knowledge. Imagine, you are given a data set from Aerospace Department, US Govt. How would you apply business logic here? In such cases, we combine levels by considering the frequency distribution or response rate.
- To combine levels using their frequency, we first look at the frequency distribution of of each level and combine levels having frequency less than 5% of total observation (5% is standard but you can change it based on distribution). This is an effective method to deal with rare levels.
- We can also combine levels by considering the response rate of each level. We can simply combine levels having similar response rate into same group.
    Finally, you can also look at both frequency and response rate to combine levels. You first combine levels based on response rate then combine rare levels to relevant group.






## Feature Engineering:
- Indicator Variables: Bining
- Interaction Features: Sum, multiply, max, stat
- Feature Representation: Encoding
- Error Analysis (Post-Modeling): Feature Selection, cross-validation
- External Data: Domain Knowledge


1. Indicator Variables

The indicator variables helps in isolate the key information.
We can help our algorithm to “focus” on what’s important by highlighting it beforehand.

    Indicator variable from thresholds: Let’s say you’re studying alcohol preferences by U.S. consumers and your dataset has an age feature. You can create an indicator variable for age >= 21 to distinguish subjects who were over the legal drinking age.
    Indicator variable from multiple features: You’re predicting real-estate prices and you have the features n_bedrooms and n_bathrooms. If houses with 2 beds and 2 baths command a premium as rental properties, you can create an indicator variable to flag them.
    Indicator variable for special events: You’re modeling weekly sales for an e-commerce site. You can create two indicator variables for the weeks of Black Friday and Christmas.
    Indicator variable for groups of classes: You’re analyzing website conversions and your dataset has the categorical feature traffic_source. You could create an indicator variable for paid_traffic by flagging observations with traffic source values of  "Facebook Ads" or "Google Adwords".

2. Interaction Features

Highlighting interactions between two or more features.(sum, difference, product, or quotient of multiple features)

    Sum of two features: Let’s say you wish to predict revenue based on preliminary sales data. You have the features sales_blue_pens and sales_black_pens. You could sum those features if you only care about overall sales_pens.
    Difference between two features: You have the features house_built_date and house_purchase_date. You can take their difference to create the feature house_age_at_purchase.
    Product of two features: You’re running a pricing test, and you have the feature price and an indicator variable conversion. You can take their product to create the feature earnings.
    Quotient of two features: You have a dataset of marketing campaigns with the features n_clicks and n_impressions. You can divide clicks by impressions to create  click_through_rate, allowing you to compare across campaigns of different volume.

3. Feature Representation

    Date and time features: Let’s say you have the feature purchase_datetime. It might be more useful to extract purchase_day_of_week and purchase_hour_of_day. You can also aggregate observations to create features such as purchases_over_last_30_days.
    Numeric to categorical mappings: You have the feature years_in_school. You might create a new feature grade with classes such as "Elementary School", "Middle School", and "High School".
    Grouping sparse classes: You have a feature with many classes that have low sample counts. You can try grouping similar classes and then grouping the remaining ones into a single "Other" class.
    Creating dummy variables: Depending on your machine learning implementation, you may need to manually transform categorical features into dummy variables. You should always do this after grouping sparse classes.

4. External Data
An underused type of feature engineering is bringing in external data. 

    Time series data: The nice thing about time series data is that you only need one feature, some form of date, to layer in features from another dataset.
    External API’s: There are plenty of API’s that can help you create features. For example, the Microsoft Computer Vision API can return the number of faces from an image.
    Geocoding: Let’s say have you street_address, city, and state. Well, you can geocode them into latitude and longitude. This will allow you to calculate features such as local demographics (e.g. median_income_within_2_miles) with the help of another dataset.
    Other sources of the same data: How many ways could you track a Facebook ad campaign? You might have Facebook’s own tracking pixel, Google Analytics, and possibly another third-party software. Each source can provide information that the others don’t track. Plus, any differences between the datasets could be informative (e.g. bot traffic that one source ignores while another source keeps).

5. Error Analysis (Post-Modeling)

Possible next steps include collecting more data, splitting the problem apart, or engineering new features that address the errors. To use error analysis for feature engineering, you’ll need to understand why your model missed its mark.

- Segment by classes (try with higher error rate, draw some indicator feature from there)
- Ask colleagues or domain experts
- Look at instaces, with higher errors. Look for patterns that you can formalize into new features.
- Unsupervised clustering


### Good features to engineer…

- Can be computed for future observations.
- Are usually intuitive to explain.
- Are informed by domain knowledge or exploratory analysis.
- Must have the potential to be predictive. Don’t just create features for the sake of it.
- Never touch the target variable. This a trap that beginners sometimes fall into. Whether you’re creating indicator variables or interaction features, never use your target variable. That’s like “cheating” and it would give you very misleading results.

---

## Feature Selection [src-analytics-vidya]:

- Filter Methods
- Wrapper Methods
- Embedded Methods
- Difference between Filter and Wrapper methods




### Filter Methods.

- Pearson’s Correlation: It is used as a measure for quantifying linear dependence between two continuous variables X and Y. Its value varies from -1 to +1.

- LDA: Linear discriminant analysis is used to find a linear combination of features that characterizes or separates two or more classes (or levels) of a categorical variable.

Chi-Square: It is a is a statistical test applied to the groups of categorical features to evaluate the likelihood of correlation or association between them using their frequency distribution.

**NOTE**: Filter Methods does not remove multicollinearity.







### wrapper methods:
Here, we try to use a subset of features and train a model using them. Based on the inferences that we draw from the previous model, we decide to add or remove features from your subset. 
-This is a search problem. 
-This is computationally very expensive.

Methods:
- forward feature selection
- backward feature elimination
- recursive feature elimination

#### Forward Selection:
Forward selection is an iterative method in which we start with having no feature in the model. In each iteration, we keep adding the feature which best improves our model till an addition of a new variable does not improve the performance of the model.

#### Backward Elimination
In backward elimination, we start with all the features and removes the least significant feature at each iteration which improves the performance of the model. We repeat this until no improvement is observed on removal of features.

#### Recursive Feature elimination
It is a greedy optimization algorithm which aims to find the best performing feature subset. It repeatedly creates models and keeps aside the best or the worst performing feature at each iteration. It constructs the next model with the left features until all the features are exhausted. It then ranks the features based on the order of their elimination.




### Difference between Filter and Wrapper methods

The main differences between the filter and wrapper methods for feature selection are:

- Filter methods measure the relevance of features by their correlation with dependent variable while wrapper methods measure the usefulness of a subset of feature by actually training a model on it.
- Filter methods are much faster compared to wrapper methods as they do not involve training the models. On the other hand, wrapper methods are computationally very expensive as well.
- Filter methods use statistical methods for evaluation of a subset of features while wrapper methods use cross validation.
- Filter methods might fail to find the best subset of features in many occasions but wrapper methods can always provide the best subset of features.
- Using the subset of features from the wrapper methods make the model more prone to overfitting as compared to using subset of features from the filter methods




## Feature Selection[More-Info](https://www.kaggle.com/kanncaa1/feature-selection-and-data-visualization)

1) Feature selection with correlation and random forest classification¶


#### correlation map
```python
f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(x.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
```

using this coorelation map, we select some of the feature and check our algo pred rate.


```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score,confusion_matrix
from sklearn.metrics import accuracy_score

# split data train 70 % and test 30 %
x_train, x_test, y_train, y_test = train_test_split(x_1, y, test_size=0.3, random_state=42)

#random forest classifier with n_estimators=10 (default)
clf_rf = RandomForestClassifier(random_state=43)
clr_rf = clf_rf.fit(x_train,y_train)

ac = accuracy_score(y_test,clf_rf.predict(x_test))
print('Accuracy is: ',ac)
cm = confusion_matrix(y_test,clf_rf.predict(x_test))
sns.heatmap(cm,annot=True,fmt="d")

Accuracy is:  0.9532163742690059
```





2) Univariate feature selection and random forest classification
n univariate feature selection, we will use SelectKBest that removes all but the k highest scoring features


```python
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
# find best scored 5 features
select_feature = SelectKBest(chi2, k=5).fit(x_train, y_train)

print('Score list:', select_feature.scores_)
print('Feature list:', x_train.columns)

```
Using this selction score, we obtain the top k feature, using `transform` function

```python
x_train_2 = select_feature.transform(x_train)
x_test_2 = select_feature.transform(x_test)
#random forest classifier with n_estimators=10 (default)
clf_rf_2 = RandomForestClassifier() 
clr_rf_2 = clf_rf_2.fit(x_train_2,y_train)
ac_2 = accuracy_score(y_test,clf_rf_2.predict(x_test_2))
print('Accuracy is: ',ac_2)
cm_2 = confusion_matrix(y_test,clf_rf_2.predict(x_test_2))
sns.heatmap(cm_2,annot=True,fmt="d")

Accuracy is:  0.9590643274853801
```





3) Recursive feature elimination (RFE) with random forest
Basically, it uses one of the classification methods (random forest in our example), assign weights to each of features. Whose absolute weights are the smallest are pruned from the current set features. That procedure is recursively repeated on the pruned set until the desired number of features

```python
from sklearn.feature_selection import RFE
# Create the RFE object and rank each pixel
clf_rf_3 = RandomForestClassifier()
rfe = RFE(estimator=clf_rf_3, n_features_to_select=5, step=1)
rfe = rfe.fit(x_train, y_train)

print('Chosen best 5 feature by rfe:',x_train.columns[rfe.support_])

Chosen best 5 feature by rfe: Index(['area_mean', 'concavity_mean', 'area_se', 'concavity_worst',
       'symmetry_worst'],
      dtype='object')
```

**In this method, we select the no of feature, what if we select less no of feature than which can increase acc much greater than this.**


4) Recursive feature elimination with cross validation and random forest classification

Now we will not only find best features but we also find how many features do we need for best accuracy.



```python
from sklearn.feature_selection import RFECV

# The "accuracy" scoring is proportional to the number of correct classifications
clf_rf_4 = RandomForestClassifier() 
rfecv = RFECV(estimator=clf_rf_4, step=1, cv=5,scoring='accuracy')   #5-fold cross-validation
rfecv = rfecv.fit(x_train, y_train)

print('Optimal number of features :', rfecv.n_features_)
print('Best features :', x_train.columns[rfecv.support_])

Optimal number of features : 14
Best features : Index(['texture_mean'....]

plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
```




5) Tree based feature selection and random forest classification

Random forest choose randomly at each iteration, therefore sequence of feature importance list can change.
```python
clf_rf_5 = RandomForestClassifier()
clr_rf_5 = clf_rf_5.fit(x_train,y_train)
importances = clr_rf_5.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf_rf.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(x_train.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest

plt.figure(1, figsize=(14, 13))
plt.title("Feature importances")
plt.bar(range(x_train.shape[1]), importances[indices],
       color="g", yerr=std[indices], align="center")
plt.xticks(range(x_train.shape[1]), x_train.columns[indices],rotation=90)
plt.xlim([-1, x_train.shape[1]])
plt.show()

Feature ranking:
1. feature 1 (0.213700) ....
```



