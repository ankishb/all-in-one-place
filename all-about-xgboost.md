<!-- ---
layout: post
title: "some-thoughts-on-practical-data-science"
categories: data-science
author: "Ankish Bansal"
---
 -->

## General Rule:
### XGBoost
#### Bias-Variance Trade off
- Controlling model complexity
    `max_depth`, `min_child_weight` and `gamma`
- Robusr to noise
    `subsample`, `colsample_bytree`


# XGBoost New parameter (Advance)
- tree_method:
    hist: Fast histogram optimized approximate greedy algorithm. It uses some performance improvements such as bins caching.
- grow_policy [default= depthwise]

    Controls a way new nodes are added to the tree.
    Currently supported only if tree_method is set to hist.
    Choices: depthwise, lossguide
        depthwise: split at nodes closest to the root.
        lossguide: split at nodes with highest loss change.

- max_leaves [default=0]

    Maximum number of nodes to be added. Only relevant when grow_policy=lossguide is set.

- max_bin, [default=256]

    Only used if tree_method is set to hist.
    Maximum number of discrete bins to bucket continuous features.
    Increasing this number improves the optimality of splits at the cost of higher computation time.





### Light-GBM
- For best fit
    `num_leaves`, `min_data_in_leaf` and `max_depth`
- For faster Speed
    `bagging_fraction`, `feature_fraction` and `max_bin`
- For better Accuracy
    `num_leaves` and `max_bin`   


### For Faster Speed
- `bagging_fraction`
- `bagging_freq`
- `max_bin`
- `feature_fraction`


    Use bagging by setting `bagging_fraction` and `bagging_freq`
    Use feature sub-sampling by setting feature_fraction
    Use small max_bin


For Better Accuracy
- `max_bin`
- `learning_rate` and `num_iterations`
- `num_leaves`


    Use large max_bin (may be slower)
    Use small learning_rate with large num_iterations
    Use large num_leaves (may cause over-fitting)
    Use bigger training data
    Try dart

Deal with Over-fitting
- `max_bin`
- `num_leaves`
- `min_data_in_leaf`
- `bagging_fraction`
- `bagging_freq`
- `lambda_l1`, `lambda_l2` and `min_gain_to_split` (**Regularization**)
- `max_depth`

    Use small max_bin
    Use small num_leaves
    Use min_data_in_leaf and min_sum_hessian_in_leaf
    Use bagging by set bagging_fraction and bagging_freq
    Use feature sub-sampling by set feature_fraction
    Use bigger training data
    Try lambda_l1, lambda_l2 and min_gain_to_split for regularization
    Try max_depth to avoid growing deep tree






```python
XGBModel(max_depth=3, learning_rate=0.1, n_estimators=100,
        silent=True, objective="reg:linear", booster='gbtree',
        n_jobs=1, nthread=None, gamma=0, min_child_weight=1, max_delta_step=0,
        subsample=1, colsample_bytree=1, colsample_bylevel=1,
        reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
        base_score=0.5, random_state=0, seed=None, missing=None,
        importance_type="gain"):
```
    """Implementation of the Scikit-Learn API for XGBoost.
    Parameters
    ----------
    -verbosity [default=1]
        Verbosity of printing messages. Valid values are 0 (silent), 1 (warning), 2 (info), 3 (debug).
    -max_depth : Maximum tree depth for base learners.
    -learning_rate : Boosting learning rate (xgb's "eta")
    -n_estimators : Number of boosted trees to fit.
    -objective : string or Specify the learning task and the corresponding learning objective or
    -booster: Specify which booster to use: gbtree, gblinear or dart.
    -n_jobs : Number of parallel threads used to run xgboost.  (replaces ``nthread``)
    -gamma : Minimum loss reduction required to make a further partition on a leaf node of the tree.
    -min_child_weight : int (Minimum sum of instance weight(hessian) needed in a child)
    -max_delta_step : int
        Maximum delta step we allow each tree's weight estimation to be.
        If the value is set to 0, it means there is no constraint   
        Usually this parameter is not needed, but it might help in logistic regression when class is extremely imbalanced.
    -subsample : Subsample ratio of the training instance.[Typical values: 0.5-1]
        Lower values make the algorithm more conservative and prevents overfitting but too small values might lead to under-fitting.
    -colsample_bytree : Subsample ratio of columns when constructing each tree.
    -colsample_bylevel : Subsample ratio of columns for each split, in each level.
    -reg_alpha : L1 regularization term on weights
    -reg_lambda : L2 regularization term on weights
    -scale_pos_weight : Balancing of positive and negative weights.
    -base_score: The initial prediction score of all instances, global bias.
    -missing : float, defaults to np.nan.
    -importance_type: string, default "gain"
        The feature importance type for the feature_importances_ property: either "gain",
        "weight", "cover", "total_gain" or "total_cover".
    
    """



objective [default=reg:linear]

- reg:linear: linear regression
- reg:logistic: logistic regression
- binary:logistic: logistic regression for binary classification, output probability
- binary:hinge: hinge loss for binary classification. This makes predictions of 0 or 1, rather than producing probabilities.
- count:poisson –poisson regression for count data, output mean of poisson distribution
    max_delta_step is set to 0.7 by default in poisson regression (used to safeguard optimization)
- multi:softmax: set XGBoost to do multiclass classification using the softmax objective, you also need to set num_class(number of classes)
- rank:pairwise: Use LambdaMART to perform pairwise ranking where the pairwise loss is minimized



eval_metric [ default according to objective ]

- rmse[default] – root mean square error
- mae – mean absolute error
- logloss – negative log-likelihood
- error[default] – Binary classification error rate (0.5 threshold)
- merror – Multiclass classification error rate
- mlogloss – Multiclass logloss
- auc: Area under the curve




## Custom objective and eveluation metrics function

```python
import numpy as np
import xgboost as xgb

dtrain = xgb.DMatrix('../data/agaricus.txt.train')
dtest = xgb.DMatrix('../data/agaricus.txt.test')

# note: for customized objective function, we leave objective as default
# note: what we are getting is margin value in prediction
# you must know what you are doing
param = {'max_depth': 2, 'eta': 1, 'silent': 1}
watchlist = [(dtest, 'eval'), (dtrain, 'train')]
num_round = 2
```
# user define objective function, given prediction, return gradient and second order gradient
# this is log likelihood loss
```python
def logregobj(preds, dtrain):
    labels = dtrain.get_label()
    preds = 1.0 / (1.0 + np.exp(-preds))
    grad = preds - labels
    hess = preds * (1.0 - preds)
    return grad, hess
```

# user defined evaluation function, return a pair metric_name, result
# NOTE: when you do customized loss function, the default prediction value is margin
# this may make builtin evaluation metric not function properly
# for example, we are doing logistic loss, the prediction is score before logistic transformation
# the builtin evaluation error assumes input is after logistic transformation
# Take this in mind when you use the customization, and maybe you need write customized evaluation function
```python
def evalerror(preds, dtrain):
    labels = dtrain.get_label()
    # return a pair metric_name, result. The metric name must not contain a colon (:) or a space
    # since preds are margin(before logistic transformation, cutoff at 0)
    return 'my-error', float(sum(labels != (preds > 0.0))) / len(labels)
```
# training with customized objective, we can also do step by step training
# simply look at xgboost.py's implementation of train
```python
bst = xgb.train(param, dtrain, num_round, watchlist, obj=logregobj, feval=evalerror)
```


## General Approach for Parameter Tuning
### Control Overfitting:

- The first way is to directly control model complexity.
    This includes `max_depth`, `min_child_weight` and `gamma`.
- The second way is to add randomness to make training robust to noise.
    This includes `subsample` and `colsample_bytree`.
    You can also reduce stepsize `eta`. Remember to increase `num_round` when you do so.

### Handle Imbalanced Dataset

- Feature selection with weighted samples approach

- If you care only about the overall performance metric (AUC) of your prediction
    Balance the positive and negative weights via `scale_pos_weight`
    Use AUC for evaluation
- If you care about predicting the right probability
    In such a case, you cannot re-balance the dataset
    Set parameter `max_delta_step` to a finite number (say 1) to help convergence




#### use XGBClassifier ==> That has CV method inbuilt
```python
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search
```
Function to train and evaluate the model
```python

def modelfit(alg, dtrain, 
            predictors, 
            useTrainCV=True, 
            cv_folds=5, 
            early_stopping_rounds=50):
    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        cvresult = xgb.cv(  xgb_param, xgtrain, 
                            num_boost_round=alg.get_params()['n_estimators'], 
                            nfold=cv_folds,
                            metrics='auc', 
                            early_stopping_rounds=early_stopping_rounds, 
                            show_progress=False)
        alg.set_params(n_estimators=cvresult.shape[0])
    
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain['Disbursed'],eval_metric='auc')
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
        
    #Print model report:
    print "\nModel Report"
    print "Accuracy : %.4g" % metrics.accuracy_score(dtrain['Disbursed'].values, dtrain_predictions)
    print "AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['Disbursed'], dtrain_predprob)
                    
    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
```



1: Fix learning rate and number of estimators for tuning tree-based parameters

Select reasonable `max_depth`=3-10, `min_child_weight`=1, `gamma`=0.1-0.2, `subsample, colsample_bytree`=0.5-0.9, `scale_pos_weight = 1`. These are just for starting point.


```python
#Choose all predictors except target & IDcols
predictors = [x for x in train.columns if x not in [target, IDcol]]
xgb1 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)
modelfit(xgb1, train, predictors)
```


2: Tune max_depth and min_child_weight

```python
param_test1 = {
 'max_depth':range(3,10,2),
 'min_child_weight':range(1,6,2)
}
gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=5,
                                                     min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8, 
                                                     objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27), 

                         param_grid = param_test1, 
                         scoring='roc_auc',
                         n_jobs=4,
                         iid=False, 
                         cv=5)

gsearch1.fit(train[predictors],train[target])


modelfit(gsearch1.best_estimator_, train, predictors)
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_
```

3: Tune gamma
```python

param_test3 = {
 'gamma':[i/10.0 for i in range(0,5)]
}
gsearch3 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=4,
 min_child_weight=6, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
 param_grid = param_test3, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch3.fit(train[predictors],train[target])
gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_

```

4: Tune subsample and colsample_bytree

Try 0.6,0.7,0.8,0.9 for both to start with and reduce the step size after-wards 

```python
param_test4 = {
 'subsample':[i/10.0 for i in range(6,10)],
 'colsample_bytree':[i/10.0 for i in range(6,10)]
}

`To check more conjustedly, use 
param_test5 = {
 'subsample':[i/100.0 for i in range(75,90,5)],
 'colsample_bytree':[i/100.0 for i in range(75,90,5)]
}
`

gsearch4 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=177, max_depth=4,
 min_child_weight=6, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
 param_grid = param_test4, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch4.fit(train[predictors],train[target])
gsearch4.grid_scores_, gsearch4.best_params_, gsearch4.best_score_
```


5: Tuning Regularization Parameters

`gamma` provides a substantial way to reduce overfitting. Tune ‘reg_alpha’ and ‘reg_lambda’.

```python
param_test6 = {
 'reg_alpha':[0, 0.001, 0.005, 0.01, 0.05]
 }
gsearch5 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=177, max_depth=4,
 min_child_weight=6, gamma=0.1, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
 param_grid = param_test6, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch5.fit(train[predictors],train[target])
gsearch5.grid_scores_, gsearch5.best_params_, gsearch5.best_score_
```
6: Reducing Learning Rate and repeat steps again

'''python
xgb4 = XGBClassifier(
 learning_rate =0.01,
 n_estimators=5000,
 max_depth=4,
 min_child_weight=6,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 reg_alpha=0.005,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)
modelfit(xgb4, train, predictors)
'''

##  As I mentioned in the end, techniques like feature engineering and blending have a much greater impact than parameter tuning. For instance, I generally do some parameter tuning and then run 10 different models on same parameters but different seeds. Averaging their results generally gives a good boost to the performance of the model.



## Startified sampling

```python
from sklearn.model_selection import StratifiedKFold
X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
y = np.array([0, 0, 1, 1])
skf = StratifiedKFold(n_splits=2)
skf.get_n_splits(X, y)

print(skf)

for train_index, test_index in skf.split(X, y):
   print("TRAIN:", train_index, "TEST:", test_index)
   X_train, X_test = X[train_index], X[test_index]
   y_train, y_test = y[train_index], y[test_index]

```

## eval_result

```python
Example

param_dist = {'objective':'binary:logistic', 'n_estimators':2}

clf = xgb.XGBModel(**param_dist)

clf.fit(X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        eval_metric='logloss',
        verbose=True)

evals_result = clf.evals_result()

The variable evals_result will contain:

{'validation_0': {'logloss': ['0.604835', '0.531479']},
'validation_1': {'logloss': ['0.41965', '0.17686']}}


```







