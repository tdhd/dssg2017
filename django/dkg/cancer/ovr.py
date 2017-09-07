import warnings,json,gzip,re
import pandas as pd
import numpy as np
from time import time
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.feature_extraction.text import HashingVectorizer, CountVectorizer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
<<<<<<< HEAD
from sklearn.metrics import label_ranking_average_precision_score, make_scorer
from sklearn.ensemble import BaggingClassifier
=======
from sklearn.metrics import label_ranking_average_precision_score
>>>>>>> 69afa5d47bf3600228e8f773c6807603bfd427f2
from scipy.sparse import hstack, vstack
from cancer.active_learning.selection_strategies import SelectionStrategies
import django.conf
from scipy.special import logit


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
    authors = HashingVectorizer(n_features=2**5).fit_transform(data.A1.map(concat_list_))
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

def compute_scores(y_test, y_predicted, labelNames):
    # the scores we want to compute
    scorers = [
        precision_score,
        recall_score,
        f1_score
        ]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        metrics = {
            s.__name__ : getSortedMetrics(y_test,y_predicted,labelNames,s)
            for s in scorers
            }
    return metrics

def model_selection(X,y):
    '''
    Runs model selection, returns fitted classifier
    '''
    # turn off warnings, usually there are some labels missing in the training set
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
<<<<<<< HEAD
        base_clf = SGDClassifier(loss="log", average=True, penalty='l1')
        text_clf = OneVsRestClassifier(base_clf)
        parameters = {'estimator__alpha': (np.logspace(-7,-5,5)).tolist()}
=======
        text_clf = OneVsRestClassifier(SGDClassifier(loss="log", average=True, penalty='l1'))
        parameters = {'estimator__alpha': (np.logspace(-7,-4,4)).tolist()}
>>>>>>> 69afa5d47bf3600228e8f773c6807603bfd427f2
        # perform gridsearch to get the best regularizer
        gs_clf = GridSearchCV(text_clf, parameters, cv=2, n_jobs=-1,verbose=4)
        gs_clf.fit(X, y)
        report(gs_clf.cv_results_)
    return gs_clf.best_estimator_

<<<<<<< HEAD
def compute_active_learning_curve(X_train,y_train,X_test,y_test,X_validation, y_validation, clf,percentage_samples=[1,2,5,10,15,30,50,100],method="margin"):
=======
def compute_active_learning_curve(X_train,y_train,X_test,y_test,X_validation, y_validation, clf,percentage_samples=[1,2,5,10,15,30,50,100]):
>>>>>>> 69afa5d47bf3600228e8f773c6807603bfd427f2
    '''
    Emulate active learning with annotators:
    for a given training, test and validation set, get the validation error by
    training on training data only, then the score when trained on training and
    test data and then the increasing validation score when adding more labelled
    data, either with random selection or with active learning. The results are
    the increase in scores with the respective sampling policy
    '''
    print('Computing active learning curve:')
<<<<<<< HEAD
    alpha = clf.estimator.alpha
    clf = OneVsRestClassifier(BaggingClassifier(SGDClassifier(loss="log",alpha=alpha, average=True, penalty='l1'), n_jobs=-1,max_samples=.2)).fit(X_train, y_train)
    baseline_low = label_ranking_average_precision_score(y_validation.toarray(), clf.predict_proba(X_validation))
    clf_trained = OneVsRestClassifier(SGDClassifier(loss="log",alpha=alpha, average=True, penalty='l1'), n_jobs=-1).fit(vstack([X_train, X_test]), vstack([y_train, y_test]))
=======
    clf = OneVsRestClassifier(SGDClassifier(loss="log",alpha=clf.estimator.alpha, average=True, penalty='l1'), n_jobs=-1).fit(X_train, y_train)
    baseline_low = label_ranking_average_precision_score(y_validation.toarray(), clf.predict_proba(X_validation))
    clf_trained = OneVsRestClassifier(SGDClassifier(loss="log",alpha=clf.estimator.alpha, average=True, penalty='l1'), n_jobs=-1).fit(vstack([X_train, X_test]), vstack([y_train, y_test]))
>>>>>>> 69afa5d47bf3600228e8f773c6807603bfd427f2
    baseline_high = label_ranking_average_precision_score(y_validation.toarray(), clf_trained.predict_proba(X_validation))
    print('\tBaseline on test: {}, baseline score on train and test {}'.format(baseline_low, baseline_high))

    # score test data for active learning sorting
    label_probas = clf.predict_proba(X_test)

    # run a random sampling procedure for training with increasing amounts of labels
    random_priorities = np.random.permutation(label_probas.shape[0])

    random_learning_curve = []
    for percentage in percentage_samples:
        n_samples = int((percentage/100.) * X_test.shape[0])
        X_labelled = X_test[random_priorities[:n_samples],:]
        y_labelled = y_test[random_priorities[:n_samples],:]
<<<<<<< HEAD
        clf_current = OneVsRestClassifier(SGDClassifier(loss="log",alpha=alpha, average=True, penalty='l1'), n_jobs=-1).fit(vstack([X_train, X_labelled]), vstack([y_train, y_labelled]))
=======
        clf_current = OneVsRestClassifier(SGDClassifier(loss="log",alpha=clf.estimator.alpha, average=True, penalty='l1'), n_jobs=-1).fit(vstack([X_train, X_labelled]), vstack([y_train, y_labelled]))
>>>>>>> 69afa5d47bf3600228e8f773c6807603bfd427f2
        current_score = label_ranking_average_precision_score(y_validation.toarray(), clf_current.predict_proba(X_validation))
        print('\t(RANDOM) Trained on {} samples ({}%) from test set - reached {} ({}%)'.format(n_samples, percentage, current_score, np.round(100.0*(current_score - baseline_low)/(baseline_high-baseline_low))))
        random_learning_curve.append(current_score)

<<<<<<< HEAD
    if method == 'margin':
        # mean distance to hyperplane
        dists = abs(logit(label_probas + 1e-7)).mean(axis=1)
        # run active learning procedure for training with increasing amounts of labels
        priorities = dists.argsort()
    elif method == "qbc":
        entropies = -(label_probas * np.log(label_probas + 1e-7)).sum(axis=1)
        priorities = entropies.argsort()[::-1]
=======
    # mean distance to hyperplane
    dists = abs(logit(label_probas)).mean(axis=1)
    # run active learning procedure for training with increasing amounts of labels
    priorities = dists.argsort()
>>>>>>> 69afa5d47bf3600228e8f773c6807603bfd427f2

    active_learning_curve = []
    for percentage in percentage_samples:
        n_samples = int((percentage/100.) * X_test.shape[0])
        X_labelled = X_test[priorities[:n_samples],:]
        y_labelled = y_test[priorities[:n_samples],:]
<<<<<<< HEAD
        # clf_current = OneVsRestClassifier(SGDClassifier(loss="log",alpha=clf.estimator.alpha, average=True, penalty='l1'),n_jobs=-1).fit(vstack([X_train, X_labelled]), vstack([y_train, y_labelled]))
        clf_current = OneVsRestClassifier(BaggingClassifier(SGDClassifier(loss="log",alpha=alpha, average=True, penalty='l1'), n_jobs=-1,max_samples=.2)).fit(vstack([X_train, X_labelled]), vstack([y_train, y_labelled]))
        current_score = label_ranking_average_precision_score(y_validation.toarray(), clf_current.predict_proba(X_validation))
        print('\t(ACTIVE LEARNING) Trained on {} samples ({}%) from test set - reached {} ({}%)'.format(n_samples, percentage, current_score, np.round(100.0*(current_score - baseline_low)/(baseline_high-baseline_low))))
        active_learning_curve.append(current_score)
        # score test data for active learning sorting
        label_probas = clf_current.predict_proba(X_test)
        if method == 'margin':
            # mean distance to hyperplane
            dists = abs(logit(label_probas)).mean(axis=1)
            # run active learning procedure for training with increasing amounts of labels
            priorities = dists.argsort()
        elif method == "qbc":
            entropies = -(label_probas * np.log(label_probas + 1e-7)).sum(axis=1)
            priorities = entropies.argsort()[::-1]
=======
        clf_current = OneVsRestClassifier(SGDClassifier(loss="log",alpha=clf.estimator.alpha, average=True, penalty='l1'),n_jobs=-1).fit(vstack([X_train, X_labelled]), vstack([y_train, y_labelled]))
        current_score = label_ranking_average_precision_score(y_validation.toarray(), clf_current.predict_proba(X_validation))
        print('\t(ACTIVE LEARNING) Trained on {} samples ({}%) from test set - reached {} ({}%)'.format(n_samples, percentage, current_score, np.round(100.0*(current_score - baseline_low)/(baseline_high-baseline_low))))
        active_learning_curve.append(current_score)
>>>>>>> 69afa5d47bf3600228e8f773c6807603bfd427f2

    return active_learning_curve, random_learning_curve, baseline_low, baseline_high

def classify_cancer(X, y, labelNames):
    '''
    Runs a multilabel classification experiment
    '''
    # TODO: use pipeline with FeatureUnion
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

    clf = model_selection(X_train, y_train)

    # predict test split to evaluate model
    y_predicted = clf.predict(X_test)
    # compute Scores
    print(y_predicted.shape)
    print(clf.predict_proba(X_test))

    metrics = compute_scores(y_test, y_predicted, labelNames)
    for metric in metrics:
        print metric, metrics[metric]

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
