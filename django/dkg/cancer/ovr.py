from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.feature_extraction.text import HashingVectorizer, CountVectorizer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from scipy.sparse import hstack
import warnings,json,gzip,re
from time import time
from sklearn.model_selection import GridSearchCV
import numpy as np

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


def getSortedMetrics(true, predicted, labels, scorer):
    '''
    Scores predictions
    '''
    score = scorer(true,predicted,average=None)
    return [(labels[l],score[l]) for l in score.argsort()[::-1]]


def features_of(data):
    '''
    Load and vectorize features
    '''
    # print("Vectorizing title character ngrams")
    # titleVectorizer = HashingVectorizer(analyzer="char_wb",ngram_range=(1,4),n_features=2**15)
    # titleVects = titleVectorizer.fit_transform(df.fulltitle.fillna(""))
    print("Vectorizing primary dates")
    dates = HashingVectorizer(n_features=16).fit_transform(data.Y1.fillna(""))
    print("Vectorizing titles")
    titles = HashingVectorizer(n_features=2**8).fit_transform(data.T1.fillna(""))
    print("Vectorizing authors")
    authors = HashingVectorizer(n_features=2**8).fit_transform(data.A1.fillna(""))
    print("Vectorizing abstracts")
    abstracts = HashingVectorizer(n_features=2**15).fit_transform(data.N2.fillna(""))
    X = hstack((dates,titles,authors,abstracts))
    print("Extracted feature vectors with %d dimensions"%X.shape[-1])
    return X

def clean_kws(kws):
    kwt = kws.lower().split(",")
    kwt = filter(lambda kw: kw not in ['', 'quelle', 'ovid', 'systematisch'], kwt)
    kwt = filter(lambda kw: not kw.startswith('20'), kwt)
    return kwt

def labels_of(data, label_col):
    print(data[label_col].head())
    data[label_col] = data[label_col].apply(clean_kws)
    print(data[label_col].head())
    labelVectorizer = MultiLabelBinarizer()
    y = labelVectorizer.fit_transform(data[label_col])
    print("Vectorized %d labels in %d dimensions"%(y.shape[-1],y.shape[1]))
    return y, labelVectorizer.classes_

def classify_cancer(X, y, labelNames):
    '''
    Runs a multilabel classification experiment
    '''
    # TODO: use pipeline with FeatureUnion
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    # turn off warnings, usually there are some labels missing in the training set
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # train a classifier
        # print("Training classifier")
        # clf = OneVsRestClassifier(SGDClassifier(loss="log"))
        # param_grid = {
        #     "estimator__alpha": [1e-6],
        #     "estimator__n_iter": [5],
        #     "estimator__penalty": ['l2']
        # }
        # gridsearch = GridSearchCV(estimator=clf,param_grid=param_grid,
        #     verbose=3,n_jobs=-1,scoring="average_precision")
        # clf = gridsearch.fit(X_train, y_train)
        # report(gridsearch.cv_results_)

        # without CV
        clf = OneVsRestClassifier(SGDClassifier(alpha=1e-6,n_iter=5,loss="log"))
        clf.fit(X_train, y_train)

    # predict test split to evaluate model
    y_predicted = clf.predict(X_test)
    # compute Scores
    print(y_predicted.shape)
    print(clf.predict_proba(X_test))
    # the scores we want to compute
    scorers = [precision_score,recall_score,f1_score]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        metrics = {s.__name__:getSortedMetrics(y_test,y_predicted,labelNames,s) for s in scorers}
    for metric in metrics:
        print metric, len(metrics[metric]), metrics[metric]

    # retrain on all data
    print("refitting clf on all data")
    # clf = clf.best_estimator_.fit(X,y)
    clf = clf.fit(X,y)
    return clf
    # dump results
    # json.dump(metrics,open("multilabel_classification_metrics.json","wt"))
    # # this assumes that
    # # - the feature extraction yields exactly the same number and ordering of samples
    # # - the number of classes doesn't change
    # predictions = classifAllData.predict_proba(X)
    # predictionsDF = pd.concat([df, pd.DataFrame(np.hstack([predictions,abs(predictions-.5)]))],
    #  axis=1, ignore_index=True)
    # predCols = ["probability-%s"%c for c in labelNames]
    # marginCols = ["distToMargin-%s"%c for c in labelNames]
    # predictionsDF.columns = df.columns.tolist() + predCols + marginCols
    # predictionsDF.to_csv(gzip.open("predictions.csv.gzip","wt"))
    # return metrics,predictionsDF
