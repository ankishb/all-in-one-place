---
layout: post
---

**As you read the tutorial, after that participate in this [kaggle compitition](https://www.kaggle.com/c/spooky-author-identification/data) to analyse your skill and practice.**



1. Explore the data
	- draw `histogram`, `cross-plot` and so on
	understand the data distribution

2. Feature Engineering
	- Come up with hypothesis (with assumption) and prove your hypothesis
	- Color can be important on buying second hand car, It is better to embedded color, instead of feeding raw data of images as it is.

	- **In text data-set, length, average and other statistics of sentence can be another features**

	- In tree based model, this statistics can be helpful

	- Log(x), log(1 + x), fit poisson distribution for counting variable

	- For large categorical in a feature, mean encoding is very helpful, also it helps in converge fast. **First check its distribution or distribution before and after encoding**
3. Fit a model



## Stacking (stack net)
	- It is a meta modelling approach.
	- In the base leevl, we train week learner and then their prediction is used by another models, to get final prediction.
	- It can be thought of as a NN model, where each node is replaced by one model.

	**Process**:
		- Split the adta in K parts
		- train weak learner on each K-1 parts and holdout one part for prediction for each weak learner

		**EXP**: 
		1. We split the dataset in 4 parts. 
		2. Now, train first weak learner on 1,2,3 and predict on 4th.
		3. Train 2nd weak learner on 1,2,4 and predict on 3rd.
		4. repeat on 
		5. Now, we have prediction of eavh learner on separate hold-out and after combining all, we get prediction on entire data-set.



## Ensemble Model:
	- train multiple model and use their prediction as feature for another staged model and so on.


## Data-Leakage

If we have information or feature in training data-set, that is outside from training data-set or that features has not any coorelation with the training data distribution, that is data-leakage

In short, data-leakage made model to learn something than what we expect it to learn or master.


## How we can induce data-leakage

- on building model using cross validation, we fit entire data (train and test) for standardization, which will have known the entire distribution. But our aim is to find that distribution by training our model only on half part i.e. trainind data-set.

> Use standarization o training data-set and while testing normalize the test data with the same parameters used in training time.




## Handle Text-Data

### TF-IDF: Term Frequency – Inverse Document Frequency
It aims to convert the text documents into vector models on the basis of occurrence of words in the documents without taking considering the exact ordering. 

For Example – let say there is a dataset of N text documents, In any document “D”, TF and IDF will be defined as –
-Term Frequency (TF) – TF for a term “t” is defined as the count of a term “t” in a document “D”
-Inverse Document Frequency (IDF) – IDF for a term is defined as logarithm of ratio of total documents available in the corpus and number of documents containing the term T.

    TF . IDF – TF IDF formula gives the relative importance of a term in a corpus (list of documents), given by the following formula below

    w_{i,j} = tf_{i,j} X log(N/df_i)

        tf_{i,j} ==> number of occurence of i in j
        df_i     ==> no of documents containing i
        N        ==> total no of documents

```python
# Always start with these features. They work (almost) everytime!
tfv = TfidfVectorizer(min_df=3,  max_features=None, 
                        strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
                        ngram_range=(1, 3), use_idf=1,smooth_idf=1,sublinear_tf=1,
                        stop_words = 'english')

# Fitting TF-IDF to both training and test sets (semi-supervised learning)
tfv.fit(list(xtrain) + list(xvalid))
xtrain_tfv =  tfv.transform(xtrain) 
xvalid_tfv = tfv.transform(xvalid)
```

### CountVectorizer
**ngramrange**: ngramrange(min, max) consider all possible word cases from min to max.

Example: ngramrange(1,3)
```python
v=text.CountVectorizer(ngramrange=(1,3)) print(v.fit(["an apple a day keeps the doctor away"]).vocabulary)

Result:
{'an': 0, 'apple': 3, 'day': 7, 'keeps': 12, 'the': 15, 'doctor': 10, 'away': 6, 'an apple': 1, 'apple day': 4, 'day keeps': 8, 'keeps the': 13, 'the doctor': 16, 'doctor away': 11, 'an apple day': 2, 'apple day keeps': 5, 'day keeps the': 9, 'keeps the doctor': 14, 'the doctor away': 17}
```
Another Example
```python
ctv = CountVectorizer(analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 3), stop_words = 'english')

# Fitting Count Vectorizer to both training and test sets (semi-supervised learning)
ctv.fit(list(xtrain) + list(xvalid))
xtrain_ctv =  ctv.transform(xtrain) 
xvalid_ctv = ctv.transform(xvalid)

```

### Topic Modelling
    [Link](https://www.analyticsvidhya.com/blog/2016/08/beginners-guide-to-topic-modeling-in-python/)


### Lexicon Normalization
Exp: run, runner, running, runs are different word, where root word is **run**. So normalization of text data help to extract good features.

-Stemming: Remove words like 'ly', 'es', 's', 'ing' etc
-Lemmatization: Obtain root words like from 'multiplying' to 'multiply'
**Note**: Normalization is not much helpful, while using Deep Learning Embedding.

```python
from nltk.stem.wordnet import WordNetLemmatizer 
lem = WordNetLemmatizer()

from nltk.stem.porter import PorterStemmer 
stem = PorterStemmer()

word = "multiplying" 
lem.lemmatize(word, "v")
>> "multiply" 
stem.stem(word)
>> "multipli
```


### Noise Removal

```python
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from nltk import word_tokenize
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
```






## Classical Validation
	We generally, split our data-set into training and testing. Further from training data-set, we take some part for validation. This is classical setting. **We use K-Fold validation strategy to obtain unbiased estimate of the performance, i.e. sum of all fold's prediction / K**

**Noe that this K-Fold validation considers on training data**

## Nested Validation
> This is more robust method, **Especially in time-series dataset, where data-leakage generally occurs and affect the model performance by an enormous amount.**

The idea is that there are two loops, One is outer loop, same as classical validation step and another is inner loop, where futher training data in one step of K-Fold is divided into training and validation and The 1-Fold, which is hold for validation in outer loop, act as testing dataset.

Using nested cross-validation, we train K-models with different paraameters, and each model use grid serach to find the optimal parameters. If our model is stable, then each model will have same hyper-parameyters in the end.


## Why is Cross-Validation Different with Time Series?

When dealing with time series data, traditional cross-validation (like k-fold) should not be used for two reasons:

- Temporal Dependencies
- Arbitrary choice of Test data-set


## Nested CV method
	-Predict Second half           

	Choose any random test set and on remaining data-set, main training and validation with temporal relation

	**Not much robust**, because opf random test-set selection.

	- Forward chaining

	Maintain temporal relation between all three train, validation and test set.

	For example, we have data for 10 days.
	1. train on 1st day, validate on 2nd and test on else
	2. train on first-two, validate on third and test on else
	3. repeat.

 	This method produces many different train/test splits and the error on each split is averaged in order to compute a robust estimate of the model error.





### TfidfVectorizer
```python
Examples
--------
>>> from sklearn.feature_extraction.text import TfidfVectorizer
>>> corpus = [
...     'This is the first document.',
...     'This document is the second document.',
...     'And this is the third one.',
...     'Is this the first document?',
... ]
>>> vectorizer = TfidfVectorizer()
>>> X = vectorizer.fit_transform(corpus)
>>> print(vectorizer.get_feature_names())
['and', 'document', 'first', 'is', 'one', 'second', 'the', 'third', 'this']
>>> print(X.shape)
(4, 9)
```

        feature_extraction.text.TfidfVectorizer( input='content', 
                                                 encoding='utf-8',
                                                 decode_error='strict', 
                                                 strip_accents=None, 
                                                 lowercase=True,
                                                 preprocessor=None, 
                                                 tokenizer=None, 
                                                 analyzer='word',
                                                 stop_words=None, 
                                                 token_pattern=r"(?u)\b\w\w+\b",
                                                 ngram_range=(1, 1), 
                                                 max_df=1.0, 
                                                 min_df=1,
                                                 max_features=None, 
                                                 vocabulary=None, 
                                                 binary=False,
                                                 dtype=np.float64, 
                                                 norm='l2', 
                                                 use_idf=True, 
                                                 smooth_idf=True,
                                                 sublinear_tf=False):
    """Convert a collection of raw documents to a matrix of TF-IDF features.
    Parameters
    ----------
    decode_error : {'strict', 'ignore', 'replace'}
    strip_accents : {'ascii', 'unicode'}
        'ascii' is a fast method that only works on characters that have
        an direct ASCII mapping.
        'unicode' is a slightly slower method that works on any characters.

    preprocessor : callable or None (default)
        Override the preprocessing (string transformation) stage while
        preserving the tokenizing and n-grams generation steps.
    tokenizer : callable or None (default)
        Override the string tokenization step while preserving the
        preprocessing and n-grams generation steps.
        Only applies if ``analyzer == 'word'``.
    analyzer : string, {'word', 'char', 'char_wb'} or callable
        Whether the feature should be made of word or character n-grams.
        Option 'char_wb' creates character n-grams only from text inside
        word boundaries; n-grams at the edges of words are padded with space.
        If a callable is passed it is used to extract the sequence of features
        out of the raw, unprocessed input.
    stop_words : string {'english'}, list, or None (default)
        If a string, it is passed to _check_stop_list and the appropriate stop
        list is returned. 'english' is currently the only supported string
        value.
    token_pattern : string
        Regular expression denoting what constitutes a "token", only used
        if ``analyzer == 'word'``. The default regexp selects tokens of 2
        or more alphanumeric characters (punctuation is completely ignored
        and always treated as a token separator).
    ngram_range : tuple (min_n, max_n)
    max_df : float in range [0.0, 1.0] or int, default=1.0
        When building the vocabulary ignore terms that have a document
        frequency strictly higher than the given threshold (corpus-specific
        stop words).
    min_df : float in range [0.0, 1.0] or int, default=1
        When building the vocabulary ignore terms that have a document
        frequency strictly lower than the given threshold. 

    max_features : int or None, default=None
        If not None, build a vocabulary that only consider the top
        max_features ordered by term frequency across the corpus.
        This parameter is ignored if vocabulary is not None.
    vocabulary : Mapping or iterable, optional
        Either a Mapping (e.g., a dict) where keys are terms and values are
        indices in the feature matrix, or an iterable over terms. If not
        given, a vocabulary is determined from the input documents.
    norm : 'l1', 'l2' or None, optional
        Norm used to normalize term vectors. None for no normalization.
    use_idf : boolean, default=True
        Enable inverse-document-frequency reweighting.
    smooth_idf : boolean, default=True
        Smooth idf weights by adding one to document frequencies, as if an
        extra document was seen containing every term in the collection
        exactly once. Prevents zero divisions.
    sublinear_tf : boolean, default=False
        Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf).
    Attributes
    ----------
    vocabulary_ : dict
        A mapping of terms to feature indices.
    idf_ : array, shape (n_features)
        The inverse document frequency (IDF) vector; only defined
        if  ``use_idf`` is True.
    stop_words_ : set
        Terms that were ignored because they either:
          - occurred in too many documents (`max_df`)
          - occurred in too few documents (`min_df`)
          - were cut off by feature selection (`max_features`).
        This is only available if no vocabulary was given.

    """




















## GridSearchCV

```python
Examples
--------
>>> from sklearn import svm, datasets
>>> from sklearn.model_selection import GridSearchCV
>>> iris = datasets.load_iris()
>>> parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
>>> svc = svm.SVC(gamma="scale")
>>> clf = GridSearchCV(svc, parameters, cv=5)
>>> clf.fit(iris.data, iris.target)
```
GridSearchCV(estimator, 
            param_grid, 						
            scoring=None, 
            fit_params=None,
            n_jobs=None, 
            iid='warn', 
            refit=True, 
            cv='warn', 
            verbose=0)
    """Exhaustive search over specified parameter values for an estimator.
    Important members are fit, predict.
    GridSearchCV implements a "fit" and a "score" method.
    It also implements "predict", "predict_proba"

    Parameters
    ----------
    estimator : estimator object.
        This is assumed to implement the scikit-learn estimator interface.
        Either estimator needs to provide a ``score`` function,
        or ``scoring`` must be passed.
    param_grid : dict or list of dictionaries
        Dictionary with parameters names (string) as keys and lists of
        parameter settings to try as values.
    scoring : string, callable, list/tuple, dict or None, default: None
        A single string (see :ref:`scoring_parameter`) or a callable
        (see :ref:`scoring`) to evaluate the predictions on the test set.
        If None, the estimator's default scorer (if available) is used.
    fit_params : dict, Parameters to pass to the fit method.
    n_jobs : Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. 
    iid : boolean, default='warn'
        If True, return the average score across folds, weighted by the number
        of samples in each test set. In this case, the data is assumed to be
        identically distributed across the folds, and the loss minimized is
        the total loss per sample, and not the mean loss across the folds.
    cv : integer, to specify the number of folds in a `(Stratified)KFold` [default 3-fold]
    refit : [default=True]
        Refit an estimator using the best found parameters on the whole
        dataset.
        The refitted estimator is made available at the ``best_estimator_``
        attribute and permits using ``predict`` directly on this
        ``GridSearchCV`` instance.

    return_train_score : boolean, optional
    ...                             
    best_estimator_ : estimator or dict
    best_score_ : float
        Mean cross-validated score of the best_estimator
    best_params_ : dict
        Parameter setting that gave the best results on the hold out data.
    """

