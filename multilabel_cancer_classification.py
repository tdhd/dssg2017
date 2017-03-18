from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import HashingVectorizer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

def classify_cancer(fn = "/Users/felix/Data/master/features/features.csv"):
    '''
    Runs a multilabel classification experiment
    '''
    X,y = getFeaturesAndLabels(fn)
    # a train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8)
    # train a classifier
    classif = OneVsRestClassifier(SGDClassifier()).fit(X_train, y_train)
    # predict
    y_predicted = classif.predict(X_test)
    # the scores we want to compute
    scorers = [precision_score,recall_score,f1_score]
    # compute Scores
    metrics = {s.__name__:getSortedMetrics(y_test,y_predicted,labelVectorizer.classes_,s) for s in scorers}
    # dump results
    json.dump(metrics,open("multilabel_classification_metrics.json","wb"))

def getFeaturesAndLabels(fn):
    '''
    Load and vectorizer features and labels (vectorized using MultiLabelBinarizer)
    '''
    df = pd.read_csv(fn)
    # tokenize and binarize cancer classification labels
    labelVectorizer = MultiLabelBinarizer()
    y = labelVectorizer.fit_transform(df.classifications.apply(tokenizeCancerLabels))
    featureVectorizer = HashingVectorizer(analyzer="char_wb",ngram_range=(1,4),n_features=2**12)
    X = featureVectorizer.transform(df.fulltitle)
    return X,y

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
    return [t for t in s.strip('[]\'').split(",") if len(t)>0]
