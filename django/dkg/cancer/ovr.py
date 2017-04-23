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
    # print("Vectorizing keywords")
    # keywordVects = HashingVectorizer(n_features=2*10).fit_transform(df.searchquery_terms.str.replace('[\[\]\'\"]',""))
    # print("Vectorizing authors")
    # authorVects = HashingVectorizer(n_features=2**15).fit_transform(df.author.fillna("").str.replace('[\[\]\'\"]',""))
    print("Vectorizing abstracts")
    abstractVects = HashingVectorizer(n_features=2**15).fit_transform(data.N2.fillna(""))
    # X = hstack((titleVects,keywordVects,authorVects,abstractVects))
    X = abstractVects
    print("Extracted feature vectors with %d dimensions"%X.shape[-1])
    return X


def labels_of(data, label_col):
    # TODO: filter empty

    print(data[label_col].head())
    data[label_col] = data[label_col].apply(lambda row: row.lower().split(","))
    print(data[label_col].head())
    labelVectorizer = MultiLabelBinarizer()
    y = labelVectorizer.fit_transform(data[label_col])
    print("Vectorized %d labels in %d dimensions"%(y.shape[-1],y.shape[1]))
    return y, labelVectorizer.classes_

def classify_cancer(X, y, labelNames):
    '''
    Runs a multilabel classification experiment
    '''
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    # turn off warnings, usually there are some labels missing in the training set
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # train a classifier
        print("Training classifier")
        clf = OneVsRestClassifier(SGDClassifier(loss="log"))
        param_grid = {
            "estimator__alpha": [1e-6],
            "estimator__n_iter": [5],
            "estimator__penalty": ['l2']
        }
        # clf = OneVsRestClassifier(LogisticRegression())
        # param_grid = {
        #     "estimator__C": [1e0, 1e1, 1e2],
        #     "estimator__penalty": ['l1', 'l2']
        # }
        gridsearch = GridSearchCV(estimator=clf,param_grid=param_grid,
            verbose=3,n_jobs=8,scoring="average_precision")
        classif = gridsearch.fit(X_train, y_train)
        report(gridsearch.cv_results_)

    # predict test split to evaluate model
    y_predicted = classif.predict(X_test)
    print(y_predicted.shape)
    print(classif.predict_proba(X_test))
    # the scores we want to compute
    scorers = [precision_score,recall_score,f1_score]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # compute Scores
        metrics = {s.__name__:getSortedMetrics(y_test,y_predicted,labelNames,s) for s in scorers}
    for metric in metrics:
        print metric, len(metrics[metric]), metrics[metric]
    return classif
    # dump results
    # json.dump(metrics,open("multilabel_classification_metrics.json","wt"))
    # print(metrics)
    # print("Retraining on all data")
    # classifAllData = classif.best_estimator_.fit(X,y)
    # print("Reading data for testing model")
    # X = getFeaturesRis(fnTest)
    # df = read_ris(fnTest)
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
