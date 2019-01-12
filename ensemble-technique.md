
[src: Analytics-vidya tutorial on Ensembling]

## Diff between XGBoost and light-GBM
XGBoost: Level-wise splitting

Light-GBM: Leaf wise splitting, Bucketing (continuous variable to discrete bins), Parallel Processing

In level wise splitting, even if there is possibility of further split at some of leaf, it is avoided in XGBoost. Here comes the advantage in the Light-GBM, where that leaf is explored further.

Note: It is also the reason for over-fitting.


## Simple Ensemble Techniques

- Max Voting
- Averaging
- Weighted Averaging

### VotingClassifier
```python
model1 = tree.DecisionTreeClassifier()
model2 = KNeighborsClassifier()
model3= LogisticRegression()

model1.fit(x_train,y_train)
model2.fit(x_train,y_train)
model3.fit(x_train,y_train)

pred1=model1.predict(x_test)
pred2=model2.predict(x_test)
pred3=model3.predict(x_test)

final_pred = np.array([])
for i in range(0,len(x_test)):
    final_pred = np.append(final_pred, mode([pred1[i], pred2[i], pred3[i]]))
```

Easiest way
```python
from sklearn.ensemble import VotingClassifier
model1 = LogisticRegression(random_state=1)
model2 = tree.DecisionTreeClassifier(random_state=1)
model = VotingClassifier(estimators=[('lr', model1), ('dt', model2)], voting='hard')
model.fit(x_train,y_train)
model.score(x_test,y_test)
```



### Averaging
```python
finalpred=(pred1+pred2+pred3)/3
```

### Weighted-Araging
```python
finalpred=(pred1*0.3+pred2*0.3+pred3*0.4)
```



## Advanced Ensemble techniques
- Stacking
- Blending


### Stacking:
1. Train a model on all fold except one
2. Repeat for all other model
3. Predict the output at all foldout on each fold with their corresponding model
4. Predict on the test-dataset for all models.

For example for 2 models, we have 4 vectors of prediction, one for prediction on all training data for each model and another for prediction on test-dataset.


5. Use those 2 vectors of prediction as in input for another models.
6. Predict on those 2 vectors of test-dataset prediction for 2 models.



```python
def Stacking(model,train,y,test,n_fold):
   folds=StratifiedKFold(n_splits=n_fold,random_state=1)
   test_pred=np.empty((test.shape[0],1),float)
   train_pred=np.empty((0,1),float)
   for train_indices,val_indices in folds.split(train,y.values):
      x_train,x_val=train.iloc[train_indices],train.iloc[val_indices]
      y_train,y_val=y.iloc[train_indices],y.iloc[val_indices]

      model.fit(X=x_train,y=y_train)
      train_pred=np.append(train_pred,model.predict(x_val))
      test_pred=np.append(test_pred,model.predict(test))
    return test_pred.reshape(-1,1),train_pred

Now we’ll create two base models – decision tree and knn.

model1 = tree.DecisionTreeClassifier(random_state=1)

test_pred1 ,train_pred1=Stacking(model=model1,n_fold=10, train=x_train,test=x_test,y=y_train)

train_pred1=pd.DataFrame(train_pred1)
test_pred1=pd.DataFrame(test_pred1)

model2 = KNeighborsClassifier()

test_pred2 ,train_pred2=Stacking(model=model2,n_fold=10,train=x_train,test=x_test,y=y_train)

train_pred2=pd.DataFrame(train_pred2)
test_pred2=pd.DataFrame(test_pred2)

Create a third model, logistic regression, on the predictions of the decision tree and knn models.

df = pd.concat([train_pred1, train_pred2], axis=1)
df_test = pd.concat([test_pred1, test_pred2], axis=1)

model = LogisticRegression(random_state=1)
model.fit(df,y_train)
model.score(df_test, y_test)
```




### Blending

Blending follows the same approach as stacking but uses only a holdout (validation) set from the train set to make predictions. In other words, unlike stacking, the predictions are made on the holdout set only. The holdout set and the predictions are used to build a model which is run on the test set. Here is a detailed explanation of the blending process:

1. The train set is split into training and validation sets.
2. Model(s) are fitted on the training set.
3. The predictions are made on the validation set and the test set.
4. The validation set and its predictions are used as features to build a new model.
5. This model is used to make final predictions on the test and meta-features


```python
df_val=pd.concat([x_val, val_pred1,val_pred2],axis=1)
df_test=pd.concat([x_test, test_pred1,test_pred2],axis=1)

model = LogisticRegression()
model.fit(df_val,y_val)
model.score(df_test,y_test)
```


## Bagging:

Note: Bootstrapping is a sampling technique in which we create subsets of observations from the original dataset, with replacement. The size of the subsets is the same as the size of the original set.

- Multiple subsets are created from the original dataset, selecting observations with replacement.
- A base model (weak model) is created on each of these subsets.
- The models run in parallel and are independent of each other.
- The final predictions are determined by combining the predictions from all the models



## Boosting 
Boosting is a sequential process, where each subsequent model attempts to correct the errors of the previous model. The succeeding models are dependent on the previous model. Let’s understand the way boosting works in the below steps.

1. A subset is created from the original dataset.
2. Initially, all data points are given equal weights.
3. A base model is created on this subset.
4. This model is used to make predictions on the whole dataset.
5. Errors are calculated using the actual values and predicted values.
6. The observations which are incorrectly predicted, are given higher weights.
7. Another model is created and predictions are made on the dataset.(This model tries to correct the errors from the previous model)
8. Similarly, multiple models are created, each correcting the errors of the previous model.
9. The final model (strong learner) is the weighted mean of all the models (weak learners). Thus, the boosting algorithm combines a number of weak learners to form a strong learner. The individual models would not perform well on the entire dataset, but they work well for some part of the dataset. Thus, each model actually boosts the performance of the ensemble.




## Bagging algorithms:

- Bagging meta-estimator
- Random forest

## Boosting algorithms:

- AdaBoost
- GBM
- XGBM
- Light GBM
- CatBoost

---



4.1 Bagging meta-estimator

Bagging meta-estimator is an ensembling algorithm that can be used for both classification (BaggingClassifier) and regression (BaggingRegressor) problems. It follows the typical bagging technique to make predictions. Following are the steps for the bagging meta-estimator algorithm:

    Random subsets are created from the original dataset (Bootstrapping).
    The subset of the dataset includes all features.
    A user-specified base estimator is fitted on each of these smaller sets.
    Predictions from each model are combined to get the final result.

Code:

from sklearn.ensemble import BaggingClassifier
from sklearn import tree
model = BaggingClassifier(tree.DecisionTreeClassifier(random_state=1))
model.fit(x_train, y_train)
model.score(x_test,y_test)
0.75135135135135134

Sample code for regression problem:

from sklearn.ensemble import BaggingRegressor
model = BaggingRegressor(tree.DecisionTreeRegressor(random_state=1))
model.fit(x_train, y_train)
model.score(x_test,y_test)

Parameters used in the  algorithms:

    base_estimator:
        It defines the base estimator to fit on random subsets of the dataset.
        When nothing is specified, the base estimator is a decision tree.
    n_estimators:
        It is the number of base estimators to be created.
        The number of estimators should be carefully tuned as a large number would take a very long time to run, while a very small number might not provide the best results.
    max_samples:
        This parameter controls the size of the subsets.
        It is the maximum number of samples to train each base estimator.
    max_features:
        Controls the number of features to draw from the whole dataset.
        It defines the maximum number of features required to train each base estimator.
    n_jobs:
        The number of jobs to run in parallel.
        Set this value equal to the cores in your system.
        If -1, the number of jobs is set to the number of cores.

    random_state:
        It specifies the method of random split. When random state value is same for two models, the random selection is same for both models.
        This parameter is useful when you want to compare different models.

 
### Random Forest

Unlike bagging meta estimator, random forest randomly selects a set of features which are used to decide the best split at each node of the decision tree.

1. Random subsets are created from the original dataset (bootstrapping).
2. At each node in the decision tree, only a random set of features are considered to decide the best split.
3. A decision tree model is fitted on each of the subsets.
4. The final prediction is calculated by averaging the predictions from all decision trees.


To sum up, Random forest randomly selects data points and features, and builds multiple trees (Forest) .

```python
from sklearn.ensemble import RandomForestClassifier
model= RandomForestClassifier(random_state=1)
model.fit(x_train, y_train)
model.score(x_test,y_test)
0.77297297297297296

You can see feature importance by using model.feature_importances_ in random forest.

for i, j in sorted(zip(x_train.columns, model.feature_importances_)):
    print(i, j)

The result is as below:

ApplicantIncome 0.180924483743
...
```

For Regression problem:
```python
from sklearn.ensemble import RandomForestRegressor
model= RandomForestRegressor()
model.fit(x_train, y_train)
model.score(x_test,y_test)
```


Parameters

    n_estimators:
        It defines the number of decision trees to be created in a random forest.
        Generally, a higher number makes the predictions stronger and more stable, but a very large number can result in higher training time.
    criterion:
        It defines the function that is to be used for splitting.
        The function measures the quality of a split for each feature and chooses the best split.
    max_features :
        It defines the maximum number of features allowed for the split in each decision tree.
        Increasing max features usually improve performance but a very high number can decrease the diversity of each tree.
    max_depth:
        Random forest has multiple decision trees. This parameter defines the maximum depth of the trees.
    min_samples_split:
        Used to define the minimum number of samples required in a leaf node before a split is attempted.
        If the number of samples is less than the required number, the node is not split.
    min_samples_leaf:
        This defines the minimum number of samples required to be at a leaf node.
        Smaller leaf size makes the model more prone to capturing noise in train data.
    max_leaf_nodes:
        This parameter specifies the maximum number of leaf nodes for each tree.
        The tree stops splitting when the number of leaf nodes becomes equal to the max leaf node.
    n_jobs:
        This indicates the number of jobs to run in parallel.
        Set value to -1 if you want it to run on all cores in the system.
    random_state:
        This parameter is used to define the random selection.
        It is used for comparison between various models.

   
### AdaBoost

Adaptive boosting or AdaBoost is one of the simplest boosting algorithms. Usually, decision trees are used for modelling. Multiple sequential models are created, each correcting the errors from the last model. AdaBoost assigns weights to the observations which are incorrectly predicted and the subsequent model works to predict these values correctly.
<!-- 
Below are the steps for performing the AdaBoost algorithm:

    Initially, all observations in the dataset are given equal weights.
    A model is built on a subset of data.
    Using this model, predictions are made on the whole dataset.
    Errors are calculated by comparing the predictions and actual values.
    While creating the next model, higher weights are given to the data points which were predicted incorrectly.
    Weights can be determined using the error value. For instance, higher the error more is the weight assigned to the observation.
    This process is repeated until the error function does not change, or the maximum limit of the number of estimators is reached.
 -->
```python
#for classification problem:

from sklearn.ensemble import AdaBoostClassifier
model = AdaBoostClassifier(random_state=1)
model.fit(x_train, y_train)
model.score(x_test,y_test)
0.81081081081081086

for regression problem:

from sklearn.ensemble import AdaBoostRegressor
model = AdaBoostRegressor()
model.fit(x_train, y_train)
model.score(x_test,y_test)

```
Parameters

    base_estimators: Any Ml algo as base learner.
    n_estimators:[default value is 10]
    learning_rate:
    max_depth: maximum depth of the individual estimator.
    n_jobs: Set value to -1 for maximum processors allowed.
    random_state : To produce same results

 
### Gradient Boosting (GBM)

In GBM, each subsequent tree in series is built on the errors calculated by the previous tree.

We will use a simple example to understand the GBM algorithm. We have to predict the age of a group of people using the below data:

1. The mean age is assumed to be the predicted value for all observations in the dataset.
2. The errors are calculated using this mean prediction and actual values of age.
3. A tree model is created using the errors calculated above as target variable. Our objective is to find the best split to minimize the error.
4. The predictions by this model are combined with the predictions 1.
5. This value calculated above is the new prediction.
6.New errors are calculated using this predicted value and actual value.
7. Steps 2 to 6 are repeated till the maximum number of iterations is reached (or error function does not change).

```python
from sklearn.ensemble import GradientBoostingClassifier
model= GradientBoostingClassifier(learning_rate=0.01,random_state=1)
model.fit(x_train, y_train)
model.score(x_test,y_test)
0.81621621621621621

Sample code for regression problem:

from sklearn.ensemble import GradientBoostingRegressor
model= GradientBoostingRegressor()
model.fit(x_train, y_train)
model.score(x_test,y_test)
```

Parameters

    min_samples_split
        Defines the minimum number of samples (or observations) which are required in a node to be considered for splitting.
    min_samples_leaf
        Defines the minimum samples required in a terminal or leaf node.
        Generally, lower values should be chosen for imbalanced class problems because the regions in which the minority class will be in the majority will be very small.

    min_weight_fraction_leaf
        Similar to min_samples_leaf but defined as a fraction of the total number of observations instead of an integer.
    max_depth
    max_features: 
        No of featur for further split
        As a thumb-rule, the square root of the total number of features works great but we should check up to 30-40% of the total number of features.
 
### XGBoost

XGBoost (extreme Gradient Boosting) is an advanced implementation of the gradient boosting algorithm. XGBoost has proved to be a highly effective ML algorithm, extensively used in machine learning competitions and hackathons. XGBoost has high predictive power and is almost 10 times faster than the other gradient boosting techniques. It also includes a variety of regularization which reduces overfitting and improves overall performance. Hence it is also known as ‘regularized boosting‘ technique.

Let us see how XGBoost is comparatively better than other techniques:

Regularization:
    Standard GBM implementation has no regularisation like XGBoost.
    Thus XGBoost also helps to reduce overfitting.
Parallel Processing:
    XGBoost implements parallel processing and is faster than GBM .
    XGBoost also supports implementation on Hadoop.
High Flexibility:
    XGBoost allows users to define custom optimization objectives and evaluation criteria adding a whole new dimension to the model.
Handling Missing Values:
    XGBoost has an in-built routine to handle missing values.
Tree Pruning:
    XGBoost makes splits up to the max_depth specified and then starts pruning the tree backwards and removes splits beyond which there is no positive gain.
Built-in Cross-Validation:
    get the exact optimum number of boosting iterations in a single run.

```python

import xgboost as xgb
model=xgb.XGBClassifier(random_state=1,learning_rate=0.01)
model.fit(x_train, y_train)
model.score(x_test,y_test)
0.82702702702702702

Sample code for regression problem:

import xgboost as xgb
model=xgb.XGBRegressor()
model.fit(x_train, y_train)
model.score(x_test,y_test)
```

Parameters

    nthread
    eta

    min_child_weight
        Defines the minimum sum of weights of all observations required in a child.
        Used to control over-fitting. 
    max_depth
    max_leaf_nodes
        The maximum number of terminal nodes or leaves in a tree.
        Can be defined in place of max_depth. 
        If this is defined, GBM will ignore max_depth.
    gamma
        Gamma specifies the minimum loss reduction required to make a split.
    subsample
        Denotes the fraction of observations to be randomly sampled for each tree.
    colsample_bytree
        fraction of columns to be randomly sampled for each tree.

 
4.6 Light GBM

- Light GBM beats all the other algorithms when the dataset is extremely large. 
- Light GBM takes lesser time to run on a huge dataset.

- leaf-wise approach while other algorithms work in a level-wise approach pattern.

***Leaf-wise grwth may cause over-fitting on smaller datasets but that can be avoided by using the ‘max_depth’ parameter for learning. You can read more about Light GBM and its comparison with XGB in this article.***

```python
import lightgbm as lgb
train_data=lgb.Dataset(x_train,label=y_train)
#define parameters
params = {'learning_rate':0.001}
model= lgb.train(params, train_data, 100) 
y_pred=model.predict(x_test)
for i in range(0,185):
   if y_pred[i]>=0.5: 
   y_pred[i]=1
else: 
   y_pred[i]=0
0.81621621621621621

Sample code for regression problem:

import lightgbm as lgb
train_data=lgb.Dataset(x_train,label=y_train)
params = {'learning_rate':0.001}
model= lgb.train(params, train_data, 100)
from sklearn.metrics import mean_squared_error
rmse=mean_squared_error(y_pred,y_test)**0.5

```
Parameters

    num_iterations:
        It defines the number of boosting iterations to be performed.
    num_leaves :
        This parameter is used to set the number of leaves to be formed in a tree.
        In case of Light GBM, since splitting takes place leaf-wise rather than depth-wise, num_leaves must be smaller than 2^(max_depth), otherwise, it may lead to overfitting.
    min_data_in_leaf :
        A very small value may cause overfitting.
        It is also one of the most important parameters in dealing with overfitting.
    max_depth:
        It specifies the maximum depth or level up to which a tree can grow.
        A very high value for this parameter can cause overfitting.
    bagging_fraction:
        It is used to specify the fraction of data to be used for each iteration.
        This parameter is generally used to speed up the training.
    max_bin :
        Defines the max number of bins that feature values will be bucketed in.
        A smaller value of max_bin can save a lot of time as it buckets the feature values in discrete bins which is computationally inexpensive.

 
4.7 CatBoost

Handling categorical variables is a tedious process, especially when you have a large number of such variables. When your categorical variables have too many labels (i.e. they are highly cardinal), performing one-hot-encoding on them exponentially increases the dimensionality and it becomes really difficult to work with the dataset.


- Doesn't handle missing values
```python
from catboost import CatBoostClassifier
model=CatBoostClassifier()
categorical_features_indices = np.where(df.dtypes != np.float)[0]
model.fit(x_train,y_train,cat_features=([ 0,  1, 2, 3, 4, 10]),eval_set=(x_test, y_test))
model.score(x_test,y_test)
0.80540540540540539

Sample code for regression problem:

from catboost import CatBoostRegressor
model=CatBoostRegressor()
categorical_features_indices = np.where(df.dtypes != np.float)[0]
model.fit(x_train,y_train,cat_features=([ 0,  1, 2, 3, 4, 10]),eval_set=(x_test, y_test))
model.score(x_test,y_test)
```

Parameters

    loss_function:
        Defines the metric to be used for training.
    iterations:
        The maximum number of trees that can be built.
        The final number of trees may be less than or equal to this number.
    learning_rate:
        Defines the learning rate.
        Used for reducing the gradient step.
    border_count:
        It specifies the number of splits for numerical features.
        It is similar to the max_bin parameter.
    depth:
        Defines the depth of the trees.
    random_seed:
        This parameter is similar to the ‘random_state’ parameter we have seen previously.
        It is an integer value to define the random seed for training.

