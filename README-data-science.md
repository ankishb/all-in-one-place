# data-science-practicles

## Encoding over-view [lib for implement](http://contrib.scikit-learn.org/categorical-encoding/polynomial.html)
- `Sum`: compares the mean of the dependent variable for a given level to the overall mean of the dependent variable over all the levels. That is, it uses contrasts between each of the first k-1 levels and level k In this example, level 1 is compared to all the others, level 2 to all the others, and level 3 to all the others.
- `Polynomial`: The coefficients taken on by polynomial coding for k=4 levels are the linear, quadratic, and cubic trends in the categorical variable. The categorical variable here is assumed to be represented by an underlying, equally spaced numeric variable. Therefore, this type of encoding is used only for ordered categorical variables with equal spacing.
- `Backward Difference` the mean of the dependent variable for a level is compared with the mean of the dependent variable for the prior level. This type of coding may be useful for a nominal or an ordinal variable.

## Target Encoding
```python
means = train.groupby(col).target.mean()
train_new[col+'_mean_target'] = train_new[col].map(means)
val_new[col+'_mean_target'] = val_new[col].map(means)
```
- For highly cardinal data-set, use `cat-boost` algo, which has inbuilt category handling capacities.

## PolynomialEncoder

If we try a polynomial encoding, we get a different distribution of values used to encode the columns:
```python
encoder = ce.polynomial.PolynomialEncoder(cols=["engine_type"])
encoder.fit(obj_df, verbose=1)
encoder.transform(obj_df).iloc[:,0:7].head()
```

## BackwardDifferenceEncoder
```python
import category_encoders as ce

# Get a new clean dataframe
obj_df = df.select_dtypes(include=['object']).copy()

# Specify the columns to encode then fit and transform
encoder = ce.backward_difference.BackwardDifferenceEncoder(cols=["engine_type"])
encoder.fit(obj_df, verbose=1)

# Only display the first 8 columns for brevity
encoder.transform(obj_df).iloc[:,0:7].head()
```

## Imp key usage of pandas

[Link][https://jeffdelaney.me/blog/useful-snippets-in-pandas/]

## Label Encoding
```python
// label encoding for reviews from 0 to 5
np.where(df['Rating'] > 3, 1, 0)
```

## Find the columns with half serached name
```python
unwanted = x.columns[x.columns.str.startswith('ps_calc_')]
x.drop(unwanted,inplace=True,axis=1)
```

## pandas DataFrame: DataFrame(data=None, index=None, columns=None, dtype=None)

```python
df = pd.DataFrame(data={'col1': [1, 2], 'col2': [3, 4]})

>>> df2 = pd.DataFrame(np.random.randint(low=0, high=10, size=(5, 5)),
...                    columns=['a', 'b', 'c', 'd', 'e'], dtype=np.int8)
>>> df2
    a   b   c   d   e
0   2   8   8   3   4
1   4   2   9   0   9
2   1   0   7   8   0
3   5   1   7   1   3
4   6   0   2   4   2
```


## Replace outlier with the median of that features
```python
mean, std = users[cols[0]].mean(), users[cols[0]].std()
median = users[cols[0]].loc[(users[cols[0]] - mean).abs() < 3*std].median()
users['testing'] = np.where((users[cols[0]] - mean).abs() > 3*std, median,users[cols[0]])
plt.plot(users['testing'])
```
    
    
## Remove outliers
```python
# np.percentile(train[cols[0]],q=0.1)
q1 = users[cols[0]].quantile(0.10)
q3 = users[cols[0]].quantile(0.90)
iqr = q3-q1
# train[cols[0]][~(train[cols[0]] < (q1-1.5*iqr) | train[cols[0]] > (q1+1.5*iqr)).
#                any(axis=1)].shape
users = users[(users[cols[0]]< (q3+1.5*iqr))]
users = users[(users[cols[0]]>(q1-1.5*iqr))]
```


## Merging array in the DataFrame
    # convert array in the list first.
    First make the list into a Series:
    se = pd.Series(mylist)
    
    Then add the values to the DataFrame:
    df['new_col'] = se.values


## One-Hot Encoding
```python
a = np.array([1, 0, 3])
b = np.zeros((3, 4))
b[np.arange(3), a] = 1
b
array([[ 0.,  1.,  0.,  0.],
       [ 1.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  1.]])
       


users = np.zeros((4,3345))
users[np.arange(4), temp[:,1].astype(int)[0]] = 1 
users
```
        
        

## pd.dummy_variable
    Convert categorical variable into dummy/indicator variables
    
```python
all_data = pd.get_dummies(all_data)

pd.get_dummies(df, prefix=['col1', 'col2'])

df = pd.DataFrame({'A': ['a', 'b', 'a'], 'B': ['b', 'a', 'c'],
            'C': [1, 2, 3]})

    C  col1_a  col1_b  col2_a  col2_b  col2_c
0  1       1       0       0       1       0
1  2       0       1       1       0       0
2  3       1       0       0       0       1
```
        
## convert data-type

The input to to_numeric() is a Series or a single column of a DataFrame.

```python
s = pd.Series(["8", 6, "7.5", 3, "0.9"]) # mixed string and numeric values
s
0      8
1      6
2    7.5
3      3
4    0.9
dtype: object

pd.to_numeric(s) # convert everything to float values
0    8.0
1    6.0
2    7.5
3    3.0
4    0.9
dtype: float64
```

As you can see, a new Series is returned. Remember to assign this output to a variable or column name to continue using it:

```python
# convert Series
my_series = pd.to_numeric(my_series)

# convert column "a" of a DataFrame
df["a"] = pd.to_numeric(df["a"])

You can also use it to convert multiple columns of a DataFrame via the apply() method:

# convert all columns of DataFrame
df = df.apply(pd.to_numeric) # convert all columns of DataFrame

# convert just columns "a" and "b"
df[["a", "b"]] = df[["a", "b"]].apply(pd.to_numeric)




# convert all DataFrame columns to the int64 dtype
df = df.astype(int)

# convert column "a" to int64 dtype and "b" to complex type
df = df.astype({"a": int, "b": complex})

# convert Series to float16 type
s = s.astype(np.float16)

# convert Series to Python strings
s = s.astype(str)
```



### Another way
```python
train.astype({"Credit_History":object}).dtypes
train.groupby(cols[4])['Loan_Status'].count()
```





## Way to replace outlier and nan values with median or mean or sth else
```python 
cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']

for col in cols:
    mean, std = train[col].mean(), train[col].std()
    median = train[col].loc[(train[col] - mean).abs() < 2*std].median()
    # fill null values with median
    train[col+'_feature'] = train[col].fillna(median)
    test[col+'_feature'] = test[col].fillna(median)
    # replace outlier with median
    train[col+'_feature'] = np.where((train[col + '_feature'] - mean).abs() > 2*std, median,train[col+'_feature'])
    test[col+'_feature'] = np.where((test[col+'_feature'] - mean).abs() > 2*std, median,test[col+'_feature'])
```


## Draw subplots with hist or others for few columns
```python
cols = ['ApplicantIncome_feature', 'CoapplicantIncome_feature', 'LoanAmount_feature']
W = 16
H = 10
fig, m_axs = plt.subplots(3,7, figsize = (W,H))
# m_axs = m_axs.flatten().T
for i in range(3):
    m_axs[i,0].plot(train[cols[i]])
#     m_axs[i,1].plot(train[cols[i]])
#     sns.boxplot(train[cols[i]],  orient='h' , ax=m_axs[i,1])
    sns.distplot(train[cols[i]], ax=m_axs[i,1])
#     sns.histplot(np.log(temp[cols[i]]), ax=m_axs[i,2])
    m_axs[i,2].hist(train[cols[i]])
    m_axs[i,3].plot(test[cols[i]])
    m_axs[i,4].hist(test[cols[i]])
    m_axs[i,5].plot(np.log(1e-9+train[cols[i]]))
    waste = np.log(1e-9+train[cols[i]])
#     print(waste.shape)
    m_axs[i,6].hist(waste)

```



## apply function 
apply a function along a specific axis (meaning, either rows (axis = 1) or columns (axis = 0)) of a DataFrame


#### Timing apply on the Haversine function

 df['feature'] = df.apply(lambda row: funct(12, row['col1'],row['col2']), axis=1)



## change value of one column using 'loc'
adjusted_ratings.loc[adjusted_ratings['rating_adjusted'] == 0, 'rating_adjusted'] = 1e-8



