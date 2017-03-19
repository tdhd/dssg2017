from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import HashingVectorizer, CountVectorizer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from scipy.sparse import hstack
import warnings,json,gzip
from time import time
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import numpy as np

TRAINDATA = "/Users/felix/Data/dssg-cancer/features/features.csv"
TESTDATA = "/Users/felix/Data/dssg-cancer/features/features.csv"

def classify_cancer(fnTrain = TRAINDATA,fnTest = TESTDATA):
    '''
    Runs a multilabel classification experiment
    '''
    X,y,labelNames = getFeaturesAndLabelsFine(fnTrain, topLabels=100)
    # a train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    # turn off warnings, usually there are some labels missing in the training set
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # train a classifier
        print("Training classifier")
        clf = OneVsRestClassifier(SGDClassifier(loss="log"))
        param_grid = {
            "estimator__alpha": [1e-8,1e-5,1e-4,1e-2],
            "estimator__n_iter": [5,10,20]
        }
        gridsearch = GridSearchCV(estimator=clf,param_grid=param_grid,
            verbose=3,n_jobs=-1,scoring="average_precision")
        classif = gridsearch.fit(X_train, y_train)
        report(gridsearch.cv_results_)

    # predict test split to evaluate model
    y_predicted = classif.predict(X_test)
    # the scores we want to compute
    scorers = [precision_score,recall_score,f1_score]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # compute Scores
        metrics = {s.__name__:getSortedMetrics(y_test,y_predicted,labelNames,s) for s in scorers}
    # dump results
    json.dump(metrics,gzip.open("multilabel_classification_metrics.json","wt"))
    print("Retraining on all data")
    classifAllData = classif.best_estimator_.fit(X,y)
    print("Reading data for testing model")
    df = pd.read_csv(fnTest)
    # this assumes that
    # - the feature extraction yields exactly the same number and ordering of samples
    # - the number of classes doesn't change
    predictions = classifAllData.predict_proba(getFeatures(fnTest))
    predictionsDF = pd.concat([df, pd.DataFrame(np.hstack([predictions,abs(predictions-.5)]))],
     axis=1, ignore_index=True)
    predCols = ["probability-%s"%c for c in labelNames]
    marginCols = ["distToMargin-%s"%c for c in labelNames]
    predictionsDF.columns = df.columns.tolist() + predCols + marginCols

    return metrics,predictionsDF

def getFeatures(fn):
    '''
    Load and vectorize features
    '''
    print("Reading data for feature extraction")
    df = pd.read_csv(fn)
    print("Vectorizing title character ngrams")
    titleVectorizer = HashingVectorizer(analyzer="char_wb",ngram_range=(1,4),n_features=2**15)
    titleVects = titleVectorizer.fit_transform(df.fulltitle.fillna(""))
    print("Vectorizing keywords")
    keywordVects = CountVectorizer().fit_transform(df.searchquery_terms.str.replace('[\[\]\'\"]',""))
    print("Vectorizing authors")
    authorVects = HashingVectorizer(n_features=2**15).fit_transform(df.author.fillna("").str.replace('[\[\]\'\"]',""))
    print("Vectorizing abstracts")
    abstractVects = HashingVectorizer(n_features=2**15).fit_transform(df.abstract.fillna("").str.replace('[\[\]\'\"]',""))
    X = hstack((titleVects,keywordVects,authorVects,abstractVects))
    print("Extracted feature vectors with %d dimensions"%X.shape[-1])
    return X

def getFeaturesAndLabelsFine(fn, topLabels=400):
    '''
    Load and vectorizer features and fine grained labels (vectorized using MultiLabelBinarizer)
    '''
    print("Reading data for label extraction")
    df = pd.read_csv(fn)
    # tokenize and binarize cancer classification labels
    print("Vectorizing labels")
    labelVectorizer = MultiLabelBinarizer()
    y = labelVectorizer.fit_transform(df.classifications.str.replace('[\[\]\'\"]',"").apply(tokenizeCancerLabels))
    print("Vectorized %d labels"%y.shape[-1])
    X = getFeatures(fn)
    # compute label histogram
    labelCounts = y.sum(axis=0)
    # truncate topLabels
    topLabelsIdx = labelCounts.argsort()[-topLabels:][::-1]
    # check whether any label of is amongst topLabels or has no label at all
    valid = ((y[:, topLabelsIdx]).sum(axis=1)>0).T | (y.sum(axis=1)==0).T
    coverage = np.double(valid.sum()) / len(df)
    print("Truncating to top %d labels accounted to %0.2f (%d/%d) of data (max count: %d, min count: %d)"%(topLabels, coverage, valid.sum(), len(df), labelCounts[topLabelsIdx[0]],labelCounts[topLabelsIdx[-1]]))
    # return only rows that either are without label or have at least one of the topLabels most frequent labels
    return X.tocsr()[valid,:],y[:,topLabelsIdx][valid,:],labelVectorizer.classes_[topLabelsIdx]

def getFeaturesAndLabelsCoarse(fn):
    '''
    Load and vectorizer features and coarse grained top level labels (vectorized using MultiLabelBinarizer)
    '''
    print("Reading data for label extraction")
    df = pd.read_csv(fn)
    # tokenize and binarize cancer classification labels
    print("Vectorizing labels")
    labelVectorizer = MultiLabelBinarizer()
    y = labelVectorizer.fit_transform(df.label_top_level.str.replace('[\[\]\'\"]',"").apply(tokenizeCancerLabels))
    print("Vectorized %d labels"%y.shape[-1])
    X = getFeatures(fn)
    return X,y,labelVectorizer.classes_

def getFeaturesAndLabelsUsefulOrNot(fn):
    '''
    Load and vectorizer features and coarse grained top level labels (vectorized using MultiLabelBinarizer)
    '''
    print("Reading data for label extraction")
    df = pd.read_csv(fn)
    # tokenize and binarize cancer classification labels
    print("Vectorizing labels")
    labelVectorizer = MultiLabelBinarizer()
    y = labelVectorizer.fit_transform(df.useful.str.replace('[\[\]\'\"]',"").apply(tokenizeCancerLabels))
    print("Vectorized %d labels"%y.shape[-1])
    X = getFeatures(fn)
    return X,y,labelVectorizer.classes_

def getSortedMetrics(true, predicted, labels, scorer):
    '''
    Scores predictions
    '''
    score = scorer(true,predicted,average=None)
    return [(labels[l],score[l]) for l in score.argsort()[::-1]]

def tokenizeCancerLabels(s):
    '''
    Tokenize the label string and remove empty strings
    '''
    return [t for t in s.split(",") if len(t)>0]

# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")
