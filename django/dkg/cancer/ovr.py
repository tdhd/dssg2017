import warnings,json,gzip,re
import pandas as pd
import numpy as np
from time import time
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import HashingVectorizer, CountVectorizer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from scipy.sparse import hstack

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
    def concat_list_(values):
        if type(values) is list:
            return ','.join(values).decode('ascii', 'ignore')
        else:
            return ''
    def extract_search_terms(kw):
        if kw.startswith('quelle'):
            return kw.split(',')[1]
        else:
            return kw

    # TODO: use data.KW quelle search keywords here, too

    # print("Vectorizing title character ngrams")
    # titleVectorizer = HashingVectorizer(analyzer="char_wb",ngram_range=(1,4),n_features=2**15)
    # titleVects = titleVectorizer.fit_transform(df.fulltitle.fillna(""))
    print("Vectorizing primary dates")
    dates = HashingVectorizer(n_features=16).fit_transform(data.Y1.map(concat_list_))
    print("Vectorizing titles")
    titles = HashingVectorizer(n_features=2**8).fit_transform(data.T1.map(concat_list_))
    print("Vectorizing authors")
    authors = HashingVectorizer(n_features=2**8).fit_transform(data.A1.map(concat_list_))
    print("Vectorizing abstracts")
    abstracts = HashingVectorizer(n_features=2**15).fit_transform(data.N2.map(concat_list_))
    X = hstack((dates,titles,authors,abstracts))
    print("Extracted feature vectors with %d dimensions"%X.shape[-1])
    return X

def clean_kws(kwt):
    kwt = map(lambda kw: kw.lower(), kwt)
    # this is filtering out quelle search query terms
    kwt = filter(lambda kw: not kw.startswith('quelle,'), kwt)
    # kwt = map(extract_search_terms, kwt)
    kwt = filter(lambda kw: not kw.startswith('20'), kwt)
    return kwt

def labels_of(data, label_col, p):
    import itertools
    import pandas as pd

    print('data label head')
    print(data[label_col].head(15))
    data[label_col] = data[label_col].apply(clean_kws)

    kw = 'keyword'
    kws = pd.DataFrame({kw: list(itertools.chain(*data[label_col]))})
    kws_r = 1.0*kws.groupby(kw).agg({kw: np.size}).sort_values(kw, ascending=False)/kws.shape[0]
    print('relative kw frequencies')
    print(kws_r.head(15))
    cutoff = np.argmin(np.abs(kws_r.cumsum() - p).values) + 1
    print('label cutoff at {}/{}'.format(cutoff, kws_r.shape[0]))

    # print('kws head')
    # valid_kws = kws[:cutoff]
    # print(valid_kws)

    print('all considered keywords')
    valid_kws = list(kws_r[:cutoff].index)
    print(valid_kws)

    def filter_kws(kws, valid):
        return filter(lambda kw: kw in valid, kws)

    data[label_col] = data[label_col].map(lambda kw: filter_kws(kw, valid_kws))
    print('data label pruned')
    print(data[label_col].head(10))

    # print(data[label_col].head())
    labelVectorizer = MultiLabelBinarizer()
    y = labelVectorizer.fit_transform(data[label_col])
    print("Vectorized %d labels in %d dimensions"%(y.shape[-1],y.shape[1]))
    return y, labelVectorizer.classes_


def classify_cancer(X, y, labelNames):
    '''
    Runs a multilabel classification experiment
    '''
    # TODO: use pipeline with FeatureUnion
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
    # turn off warnings, usually there are some labels missing in the training set
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        text_clf = OneVsRestClassifier(SGDClassifier(class_weight='balanced', loss="log", n_iter=10))
        parameters = {'estimator__alpha': (10.**np.arange(-7,-2)).tolist()}
        # perform gridsearch to get the best regularizer
        gs_clf = GridSearchCV(text_clf, parameters, 'precision_weighted', cv=2, n_jobs=-1,verbose=4)
        gs_clf.fit(X_train, y_train)
        report(gs_clf.cv_results_)
        clf = gs_clf.best_estimator_

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

    # dump results
    json.dump(metrics,open("multilabel_classification_metrics.json","wt"))

    # retrain on all data
    print("refitting clf on all data")
    # clf = clf.best_estimator_.fit(X,y)
    clf = clf.fit(X,y)
    return clf
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
