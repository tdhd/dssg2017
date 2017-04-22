from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import HashingVectorizer, CountVectorizer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from scipy.sparse import hstack
import warnings,json,gzip,re
from time import time
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from labels import to_list, flat_map, classifications_list_to_cancer_paper_subs, classifications_list_to_cancer_paper_subs_pruned, paper_types_from, cancer_types_from

TRAINDATA = "/home/ppschmidt/dssg2017/data/master/features/features.csv"
TESTDATA = TRAINDATA # "/Users/felix/Data/dssg-cancer/features/features.csv"
RISDATA = "/home/ppschmidt/dssg17/hoden-reviews-ovid-update-201606-originial.ris"

RISSTART = '\n'

def read_article(lines):
    article = {}
    for line in lines:
        match = re.match("[A-Z0-9]{2}\s+-",line)
        if match:
            key,value = line[:2], line[match.span()[1]:].strip()
            if key in article: article[key] += "," + value
            else: article[key] = value
    return article

def read_ris(fn):
    lines = open(fn,"rt", errors='ignore').readlines()
    startArticle = [idx for idx,l in enumerate(lines) if re.match(RISSTART, l)]
    articles = [read_article(lines[startArticle[s]:startArticle[s+1]]) for s in range(len(startArticle)-1)]
    return pd.DataFrame(articles)

def classify_cancer(fnTrain = TRAINDATA,fnTest = RISDATA):
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
            "estimator__alpha": [1e-6,1e-5],
            "estimator__n_iter": [20,40,50,60],
            "estimator__penalty": ['l1', 'l2']
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
    json.dump(metrics,open("multilabel_classification_metrics.json","wt"))
    print(metrics)
    print("Retraining on all data")
    classifAllData = classif.best_estimator_.fit(X,y)
    print("Reading data for testing model")
    X = getFeaturesRis(fnTest)
    df = read_ris(fnTest)
    # this assumes that
    # - the feature extraction yields exactly the same number and ordering of samples
    # - the number of classes doesn't change
    predictions = classifAllData.predict_proba(X)
    predictionsDF = pd.concat([df, pd.DataFrame(np.hstack([predictions,abs(predictions-.5)]))],
     axis=1, ignore_index=True)
    predCols = ["probability-%s"%c for c in labelNames]
    marginCols = ["distToMargin-%s"%c for c in labelNames]
    predictionsDF.columns = df.columns.tolist() + predCols + marginCols
    predictionsDF.to_csv(gzip.open("predictions.csv.gzip","wt"))
    return metrics,predictionsDF

def getFeaturesRis(fn):
    '''
    Load and vectorize features
    '''
    print("Reading data for feature extraction")
    df = read_ris(fn)
    print("Vectorizing title character ngrams")
    titleVectorizer = HashingVectorizer(analyzer="char_wb",ngram_range=(1,4),n_features=2**15)
    titleVects = titleVectorizer.fit_transform(df.T1.fillna(""))
    print("Vectorizing keywords")
    keywordVects = HashingVectorizer(n_features=2*10).fit_transform(df.KW.str.replace('[\[\]\'\"]',""))
    print("Vectorizing authors")
    authorVects = HashingVectorizer(n_features=2**15).fit_transform(df.A1.fillna("").str.replace('[\[\]\'\"]',""))
    print("Vectorizing abstracts")
    abstractVects = HashingVectorizer(n_features=2**15).fit_transform(df.N2.fillna("").str.replace('[\[\]\'\"]',""))
    X = hstack((titleVects,keywordVects,authorVects,abstractVects))
    print("Extracted feature vectors with %d dimensions"%X.shape[-1])
    return X

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
    keywordVects = HashingVectorizer(n_features=2*10).fit_transform(df.searchquery_terms.str.replace('[\[\]\'\"]',""))
    print("Vectorizing authors")
    authorVects = HashingVectorizer(n_features=2**15).fit_transform(df.author.fillna("").str.replace('[\[\]\'\"]',""))
    print("Vectorizing abstracts")
    abstractVects = HashingVectorizer(n_features=2**15).fit_transform(df.abstract.fillna("").str.replace('[\[\]\'\"]',""))
    X = hstack((titleVects,keywordVects,authorVects,abstractVects))
    print("Extracted feature vectors with %d dimensions"%X.shape[-1])
    return X

def getFeaturesAndLabelsFine(fn, topLabels=100):
    '''
    Load and vectorizer features and fine grained labels (vectorized using MultiLabelBinarizer)
    '''
    print("Reading data for label extraction")
    df = pd.read_csv(fn)
    # tokenize and binarize cancer classification labels
    print("Vectorizing labels")
    labelVectorizer = MultiLabelBinarizer()
    df['classifications_list'] = df.apply(lambda row: to_list(row, 'classifications'), axis=1)
    #df['cancer_paper_subs'] = df['classifications_list'].apply(classifications_list_to_cancer_paper_subs)
    #df['paper_types'] = df['classifications_list'].apply(paper_types_from)
    #df['cancer_types'] = df['classifications_list'].apply(cancer_types_from)
    df['cancer_paper_subs_pruned'] = df['classifications_list'].apply(classifications_list_to_cancer_paper_subs_pruned)
    y = labelVectorizer.fit_transform(df['cancer_paper_subs_pruned'])
    print("Vectorized %d labels in %d dimensions"%(y.shape[-1],y.shape[1]))
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
