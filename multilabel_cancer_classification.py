from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.feature_extraction.text import HashingVectorizer
import pandas as pd
from sklearn.model_selection import train_test_split

def classify_cancer(fn):
    fn = "/Users/felix/Data/master/features/features.csv"
    df = pd.read_csv(fn)
    # tokenize and binarize cancer classification labels
    labelVectorizer = MultiLabelBinarizer()
    y = labelVectorizer.fit_transform(df.classifications.apply(tokenizeCancerLabels))
    featureVectorizer = HashingVectorizer(analyzer="char_wb",ngram_range=(1,4),n_features=2**12)
    X = featureVectorizer.transform(df.fulltitle)
    # a classifier
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8)

    classif = OneVsRestClassifier(SVC(kernel='linear')).fit_transform(X_train, y_train)



def tokenizeCancerLabels(s):
    return [t for t in s.strip('[]\'').split(",") if len(t)>0]
