


## stratify sampling or splitting       
    ```python
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(df.col1,df.target,
                                                        stratify=df.target, 
                                                        test_size=0.2) 
    ```

## ensemble method ==> RandomForest (to select the important features)
### (4459, 4730)   ===>>   (4459, 1000)

```python
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    NUM_OF_FEATURES = 1000

    x1, x2, y1, y2 = train_test_split(
        train, y_train.values, test_size=0.20, random_state=5)
    model = RandomForestRegressor(n_jobs=-1, random_state=7)
    model.fit(x1, y1)

    df = pd.DataFrame({'importance': model.feature_importances_, 'feature': train.columns}).sort_values(
        by=['importance'], ascending=[False])[:NUM_OF_FEATURES]

    col = df['feature'].values
    train = train[col]
    test = test[col]
    train.shape

    (4459, 1000)
```





## Check the train-test data-set distribution using **scipy.stats.ks_2samp**
> If a feature has different distributions in training set than in testing set, we should remove this feature since what we learned during training cannot generalize.


### scipy.stats.ks_2samp(data1, data2)[source]
    """Computes the Kolmogorov-Smirnov statistic on 2 samples.

    This is a two-sided test for the null hypothesis that 2 independent samples are drawn from the same continuous distribution.

    Returns:	
        D : KS statistic
        p-value : wo-tailed p-value
    """

"""
But if the statistic is small or the p-value is high, then we can reject the null hypothesis.**because distributions of the two samples are the same** So it is more correct to write: 
if pvalue <= THRESHOLD_P_VALUE or np.abs(statistic) > THRESHOLD_STATISTIC
"""

```python

from scipy.stats import ks_2samp
THRESHOLD_P_VALUE = 0.01 #need tuned
THRESHOLD_STATISTIC = 0.3 #need tuned
diff_cols = []
for col in train.columns:
    statistic, pvalue = ks_2samp(train[col].values, test[col].values)
    if pvalue <= THRESHOLD_P_VALUE and np.abs(statistic) > THRESHOLD_STATISTIC:
        diff_cols.append(col)
for col in diff_cols:
    if col in train.columns:
        train.drop(col, axis=1, inplace=True)
        test.drop(col, axis=1, inplace=True)
train.shape

(4459, 1000)
```



## Dimensioality Reduction by preserving pairwise distances between samples

**Best for distance based method**
Two method:
1. sklearn.random_projection.GaussianRandomProjection 
2. sklearn.random_projection.SparseRandomProjection

```python
from sklearn.random_projection import SparseRandomProjection
ntrain = len(train)
ntest = len(test)
tmp = pd.concat([train,test])#RandomProjection


NUM_OF_COM = 100 #need tuned
transformer = SparseRandomProjection(n_components = NUM_OF_COM)
RP = transformer.fit_transform(tmp)
rp = pd.DataFrame(RP)
columns = ["RandomProjection{}".format(i) for i in range(NUM_OF_COM)]
rp.columns = columns

rp_train = rp[:ntrain]
rp_test = rp[ntrain:]
rp_test.index = test.index

#concat RandomProjection and raw data
train = pd.concat([train,rp_train],axis=1)
test = pd.concat([test,rp_test],axis=1)
```



## Feature Engineering
### Adding Statistics of features such as **weight_count, count_not0, sum, var, median, mean, std, min, max, skew, kurtosis**can be helpful in some cases:

```python
weight = ((train != 0).sum()/len(train)).values
tmp_train = train[train!=0]
tmp_test = test[test!=0]
train["weight_count"] = (tmp_train*weight).sum(axis=1)
test["weight_count"] = (tmp_test*weight).sum(axis=1)
train["count_not0"] = (train != 0).sum(axis=1)
test["count_not0"] = (test != 0).sum(axis=1)
train["sum"] = train.sum(axis=1)
test["sum"] = test.sum(axis=1)
train["var"] = tmp_train.var(axis=1)
test["var"] = tmp_test.var(axis=1)
train["median"] = tmp_train.median(axis=1)
test["median"] = tmp_test.median(axis=1)
train["mean"] = tmp_train.mean(axis=1)
test["mean"] = tmp_test.mean(axis=1)
train["std"] = tmp_train.std(axis=1)
test["std"] = tmp_test.std(axis=1)
train["max"] = tmp_train.max(axis=1)
test["max"] = tmp_test.max(axis=1)
train["min"] = tmp_train.min(axis=1)
test["min"] = tmp_test.min(axis=1)
train["skew"] = tmp_train.skew(axis=1)
test["skew"] = tmp_test.skew(axis=1)
train["kurtosis"] = tmp_train.kurtosis(axis=1)
test["kurtosis"] = tmp_test.kurtosis(axis=1)
del(tmp_train)
del(tmp_test)
```

        
        
        


        
        
        
## Before Features engineering, always do this
lets first concatenate the train and test data in the same dataframe

```python
ntrain = train.shape[0]
ntest = test.shape[0]
y_train = train.target.values
all_data = pd.concat((train, test)).reset_index(drop=True)
all_data.drop(['target'], axis=1, inplace=True)           
print("all_data size is : {}".format(all_data.shape))
```



## Missing Data
It nicely find the ratio of missing element, and show in descening order

```python
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
missing_data.head(20)
```






## Transforming some numerical variables that are really categorical
```python
#MSSubClass=The building class
all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)


#Changing OverallCond into a categorical variable
all_data['OverallCond'] = all_data['OverallCond'].astype(str)
all_data['YrSold'] = all_data['YrSold'].astype(str)
all_data['MoSold'] = all_data['MoSold'].astype(str)
```


## Skewed Features

### Skewed features FiNDING using **kurtosis, skew**
To check skewness in data, use **plt.hist(data.col1, bins='auto')**

```python
import numpy as np
from scipy.stats import kurtosis, skew

x = np.random.normal(0, 2, 10000)   # create random values based on a normal distribution

print( 'excess kurtosis of normal distribution (should be 0): {}'.format( kurtosis(x) ))
print( 'skewness of normal distribution (should be 0): {}'.format( skew(x) ))
```



### Skewed features handling
The Box-Cox transformation computed by boxcox1p is:

    y = ((1+x)**lmbda - 1) / lmbda  if lmbda != 0
        log(1+x)                    if lmbda == 0



### Complete example of SKEWED FEATURE HANDLING
```python
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

# Check the skew of all numerical features
skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness.head(10)

skewness = skewness[abs(skewness) > 0.75]
print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))
```
```python
from scipy.special import boxcox1p
skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    #all_data[feat] += 1
    all_data[feat] = boxcox1p(all_data[feat], lam)

#all_data[skewed_features] = np.log1p(all_data[skewed_features])

```


# Very Imp step:  how to do modeling in very efficient way
```python

        from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
        from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
        from sklearn.kernel_ridge import KernelRidge
        from sklearn.pipeline import make_pipeline
        from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
        from sklearn.model_selection import KFold, cross_val_score, train_test_split
        from sklearn.metrics import mean_squared_error
        import xgboost as xgb
        import lightgbm as lgb

        Define a cross validation strategy

        We use the cross_val_score function of Sklearn. However this function has not a shuffle attribut, we add then one line of code, in order to shuffle the dataset prior to cross-validation

        #Validation function
        n_folds = 5

        def rmsle_cv(model):
            kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
            rmse= np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv = kf))
            return(rmse)

        Base models

            LASSO Regression : very sensitive to outliers. So we need to made it more robust on them. 

        # Before using this, remove all outlier, either by your own by taking ist and 3rd quantile or 
        # or 
        from sklearn.preprocessing import RobustScaler
        lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))


        lasso = Lasso(alpha =0.0005, random_state=1)

            Elastic Net Regression : again made robust to outliers

        ENet = ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3)

            Kernel Ridge Regression :

        KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)

            Gradient Boosting Regression :

        With huber loss that makes it robust to outliers

        GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                           max_depth=4, max_features='sqrt',
                                           min_samples_leaf=15, min_samples_split=10, 
                                           loss='huber', random_state =5)

            XGBoost :

        model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                                     learning_rate=0.05, max_depth=3, 
                                     min_child_weight=1.7817, n_estimators=2200,
                                     reg_alpha=0.4640, reg_lambda=0.8571,
                                     subsample=0.5213, silent=1,
                                     random_state =7, nthread = -1)

            LightGBM :

        model_lgb = lgb.LGBMRegressor(objective='regression',
                                    num_leaves=5,
                                    learning_rate=0.05, n_estimators=720,
                                    max_bin = 55, bagging_fraction = 0.8,
                                    bagging_freq = 5, feature_fraction = 0.2319,
                                    feature_fraction_seed=9, bagging_seed=9,
                                    min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)

        Base models scores

        Lets see how these base models perform on the data by evaluating the cross-validation rmsle error

        score = rmsle_cv(lasso)
        print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

        Lasso score: 0.1115 (0.0074)

        score = rmsle_cv(ENet)
        print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

        ElasticNet score: 0.1116 (0.0074)

        score = rmsle_cv(KRR)
        print("Kernel Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

        Kernel Ridge score: 0.1153 (0.0075)

        score = rmsle_cv(GBoost)
        print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

        Gradient Boosting score: 0.1177 (0.0080)

        score = rmsle_cv(model_xgb)
        print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

        Xgboost score: 0.1161 (0.0079)

        score = rmsle_cv(model_lgb)
        print("LGBM score: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))

        LGBM score: 0.1157 (0.0067
```





[StackingAveragingModel to introduce Meta-Model][https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard]




from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb



#define evaluation method for a given model. we use k-fold cross validation on the training set. 
#the loss function is root mean square logarithm error between target and prediction
#note: train and y_train are feeded as global variables
NUM_FOLDS = 5 #need tuned
def rmsle_cv(model):
    kf = KFold(NUM_FOLDS, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse= np.sqrt(-cross_val_score(model, train, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)




model_xgb = xgb.XGBRegressor(colsample_bytree=0.055, colsample_bylevel =0.5, 
                             gamma=1.5, learning_rate=0.02, max_depth=32, 
                             objective='reg:linear',booster='gbtree',
                             min_child_weight=57, n_estimators=1000, reg_alpha=0, 
                             reg_lambda = 0,eval_metric = 'rmse', subsample=0.7, 
                             silent=1, n_jobs = -1, early_stopping_rounds = 14,
                             random_state =7, nthread = -1)
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=144,
                              learning_rate=0.005, n_estimators=720, max_depth=13,
                              metric='rmse',is_training_metric=True,
                              max_bin = 55, bagging_fraction = 0.8,verbose=-1,
                              bagging_freq = 5, feature_fraction = 0.9) 
score = rmsle_cv(model_xgb)
print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(model_lgb)
print("LGBM score: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))









## LGB use

        from sklearn import model_selection, preprocessing, metrics
        import lightgbm as lgb


        def run_lgb(train_X, train_y, val_X, val_y, test_X):
            params = {
                "objective" : "regression",
                "metric" : "rmse",
                "num_leaves" : 30,
                "min_child_weight" : 50,
                "learning_rate" : 0.05,
                "bagging_fraction" : 0.7,
                "feature_fraction" : 0.7,
                "bagging_frequency" : 5,
                "bagging_seed" : 2018,
                "verbosity" : -1
            }

            lgtrain = lgb.Dataset(train_X, label=train_y)
            lgval = lgb.Dataset(val_X, label=val_y)
            evals_result = {}
            model = lgb.train(params, lgtrain, 1000, valid_sets=[lgval], early_stopping_rounds=100, verbose_eval=100, evals_result=evals_result)

            pred_test_y = model.predict(test_X, num_iteration=model.best_iteration)
            return pred_test_y, model, evals_result

        train_X = train_df[cols_to_use]
        test_X = test_df[cols_to_use]
        train_y = train_df[target_col].values

        pred_test = 0
        kf = model_selection.KFold(n_splits=5, random_state=2018, shuffle=True)
        for dev_index, val_index in kf.split(train_df):
            dev_X, val_X = train_X.loc[dev_index,:], train_X.loc[val_index,:]
            dev_y, val_y = train_y[dev_index], train_y[val_index]

            pred_test_tmp, model, evals_result = run_lgb(dev_X, dev_y, val_X, val_y, test_X)
            pred_test += pred_test_tmp
        pred_test /= 5.



        fig, ax = plt.subplots(figsize=(12,10))
        lgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)
        ax.grid(False)
        plt.title("LightGBM - Feature Importance", fontsize=15)
        plt.show()
        
