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


def classify_cancer(fn = "/Users/felix/Data/master/features/features.csv"):
    '''
    Runs a multilabel classification experiment
    '''
    X,y,labelNames = getFeaturesAndLabels(fn)
    # a train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    # turn off warnings, usually there are some labels missing in the training set
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # train a classifier
        print("Training classifier")
        classif = OneVsRestClassifier(SGDClassifier(n_jobs=-1)).fit(X_train, y_train)
    # predict
    y_predicted = classif.predict(X_test)
    # the scores we want to compute
    scorers = [precision_score,recall_score,f1_score]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # compute Scores
        metrics = {s.__name__:getSortedMetrics(y_test,y_predicted,labelNames,s) for s in scorers}
    # dump results
    json.dump(metrics,gzip.open("multilabel_classification_metrics.json","w"))
    return metrics

def getFeaturesAndLabels(fn):
    '''
    Load and vectorizer features and labels (vectorized using MultiLabelBinarizer)
    '''
    print("Reading data")
    df = pd.read_csv(fn)
    # tokenize and binarize cancer classification labels
    print("Vectorizing labels")
    labelVectorizer = MultiLabelBinarizer()
    y = labelVectorizer.fit_transform(df.classifications.str.replace('[\[\]\'\"]',"").apply(tokenizeCancerLabels))
    print("Vectorized %d labels"%y.shape[-1])
    print("Vectorizing title character ngrams")
    titleVectorizer = HashingVectorizer(analyzer="char_wb",ngram_range=(1,4),n_features=2**12)
    titleVects = titleVectorizer.fit_transform(df.fulltitle.fillna(""))
    print("Vectorizing keywords")
    keywordVects = CountVectorizer().fit_transform(df.searchquery_terms.str.replace('[\[\]\'\"]',""))
    print("Vectorizing authors")
    authorVects = HashingVectorizer(n_features=2**12).fit_transform(df.author.fillna("").str.replace('[\[\]\'\"]',""))
    print("Vectorizing abstracts")
    abstractVects = HashingVectorizer(n_features=2**12).fit_transform(df.abstract.fillna("").str.replace('[\[\]\'\"]',""))
    X = hstack((titleVects,keywordVects,authorVects,abstractVects))
    print("Extracted feature vectors with %d dimensions"%X.shape[-1])
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
