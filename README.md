# all-in-one-place



# Feature extraction

- [Categorical encoding by contrast Coding]
- Frequency Encoding
- [Feature Engineering]
	- Indicator Variables: Bining
	- Interaction Features: Sum, multiply, max, stat
	- Feature Representation: Encoding
	- Error Analysis (Post-Modeling): Feature Selection, cross-validation
	- External Data: Domain Knowledge
- [Feature Selection]
	- Filter Methods
	- Wrapper Methods
	- Embedded Methods
	- Difference between Filter and Wrapper methods
- [Feature Selection] kaggle
	1) Feature selection with correlation and random forest classification¶
	2) Univariate feature selection and random forest classification
	3) Recursive feature elimination (RFE) with random forest
	4) Recursive feature elimination with cross validation and random forest classification
	5) Tree based feature selection and random forest classification


# practical-data-science

- Stacking (stack net)
- Ensemble Model:
- Data-Leakage
- Handle Text-Data
- TF-IDF
- CountVectorizer
- Topic Modelling
- Lexicon Normalization
- Noise Removal in text -stop-words-
- Classical Validation
- Nested Validation
- Why is Cross-Validation Different with Time Series?

- TfidfVectorizer -example-docs
- GridSearchCV -example-docs



# temp.md [full of practical exp]

- stratify sampling or splitting
- Feature Selection using RandomForest(ensemble method)
- Distribution Check
- scipy.stats.ks_2samp(data1, data2)
- Dimensioality Reduction by preserving pairwise distances between samples **Best for distance based method**
	1. `GaussianRandomProjection`
	2. `SparseRandomProjection`

- Feature Engineering(Stat)
- Check all Missing Data
- Transforming some numerical variables that are really categorical
- Handling Skewed Features **Box-Cox transformation**
- Base models -examples-with-cv
	- LASSO Regression 
	- Elastic Net Regression 
	- Gradient Boosting Regression 
	- KernelRidge
	- XGB
	- GBM
	- LGB

- StackingAveragingModel -link-
- LGB model -full example-




# ensemble-technique (complete exp and parameters list)

- Diff between XGBoost and light-GBM
- Simple Ensemble Techniques
	- Max Voting
	- Averaging
	- Weighted Averaging
- Advanced Ensemble techniques
	- Stacking
	- Blending
- Bagging:
- Boosting 
- Bagging algorithms:
	- Bagging meta-estimator
	- Random forest

- Boosting algorithms:
	- AdaBoost
	- GBM
	- XGBM
	- Light GBM
	- CatBoost



# all-about-xgboost

- XGBModel (complete doc)
- Custom objective and eveluation metrics function
- General Approach for Parameter Tuning
	- Control Overfitting:
	- Handle Imbalanced Dataset
- use XGBClassifier ==> That has CV method inbuilt

	1. Fix learning rate and number of estimators for tuning tree-based parameters
	2. Tune max_depth and min_child_weight
	3. Tune gamma
	4. Tune subsample and colsample_bytree
	5. Tuning Regularization Parameters
	6. Reducing Learning Rate and repeat steps again

- Startified sampling
- eval_result








# Matrplotlib-hacks
- Some-Style-Background
- Seaborn plots
	- Violin plot
	- Joint plot
	- Swarm plot
	- Pair Grid plot









# Keras-short-docs
- verbose
- ModelCheckpoint
- EarlyStopping
- ReduceLROnPlateau
- fit
- fit_generator
- Custorm Training -complete-training
- History object
- LSTM
	- return-sequence
	- backword
	- bi-direction
- TimeDistributed
- Stratify k-Fold training
- TensorBoard


# keras-another-docs
- Usuage of regularizer(#usuage-of-regularizer)
- Custom Loss function(#custom-loss-function)
- Usuage of initializers(#usuage-of-initializers)
- Keras Models(#keras-models)
- Usuage of optimizers(#usuage-of-optimizers)
    - SGD
    - RMSPRop
    - Adagrad
    - AdaDelta
    - Adam
    - Nadam



# text-handling

- SVD
- Create the pipeline with gridsearch
- Word-Vectors
	- load the GloVe vectors in a dictionary:
	- tnormalized vector for the whole sentence
- Using all keras for text handling
	- tokenizer
	- pad-seq
	- load embedding(from trained glove model)
	- model to train





# data-science-tricks

- Imp key usage of padas -link-
- Find the columns with half serached name
- pandas DataFrame
- Replace outlier with the median of that features
- Remove outliers
- Merging array in the DataFrame
- One-Hot Encoding
- pd.dummy_variable
- convert data-type
- Way to replace outlier and nan values with median or mean or sth else
- Draw subplots with hist or others for few columns
- apply function 
- change value of one column using 'loc'
