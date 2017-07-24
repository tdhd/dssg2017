import sklearn.feature_extraction
import sklearn.preprocessing
import scipy.sparse
from cancer.models import all_articles_by
import cancer.ovr
import cancer.models
from sklearn.externals import joblib
import cPickle


def encode_features_of(articles_with_keywords_and_probas):
    """
    returns scipy.sparse matrix from model articles features.

    :param articles_with_keywords_and_probas: pandas.DataFrame.
    :return: sparse design matrix.
    """
    abstracts = sklearn.feature_extraction.text.HashingVectorizer(n_features=2**15)
    X_abstracts = abstracts.fit_transform(articles_with_keywords_and_probas.abstract)

    titles = sklearn.feature_extraction.text.HashingVectorizer(n_features=2**8)
    X_titles = titles.fit_transform(articles_with_keywords_and_probas.title)

    X = scipy.sparse.hstack((X_titles, X_abstracts))
    return X


def encode_labels_of(articles_with_keywords_and_probas):
    """
    returns scipy.sparse matrix from model article labels.
    :param articles_with_keywords_and_probas: pandas.DataFrame
    :return:
    """
    def preprocess_keywords(keywords):
        """
        Lower-case keyword tokens and remove dates and search sources keywords.

        :param keywords: pandas.Series of (kw, proba, annotator)
        :return: cleaned list of string keywords
        """
        keywords_filtered = keywords.apply(lambda kws: [kw[0].lower() for kw in kws])
        keywords_filtered = keywords_filtered.apply(lambda kws: [kw for kw in kws if not kw.startswith('quelle,') and not kw.startswith('20')])
        return keywords_filtered

    def prune_keywords(keywords, p):
        """
        removes keywords that occur too infrequently.

        :param keywords: pandas.Serioes of lists with article keywords.
        :param p: cumsum relative keyword occurrence cutoff.
        :return:
        """
        import itertools
        import pandas as pd
        import numpy as np

        kws = pd.DataFrame({'kw': list(itertools.chain(*keywords))})
        print 'kws head', kws.head()

        kws_r = 1.0*kws.groupby('kw').agg({'kw': np.size}).sort_values('kw', ascending=False)/kws.shape[0]
        print('relative kw frequencies')
        print(kws_r.head(15))
        cutoff = np.argmin(np.abs(kws_r.cumsum() - p).values) + 1
        print('label cutoff at {}/{}'.format(cutoff, kws_r.shape[0]))

        print('all considered keywords')
        valid_kws = list(kws_r[:cutoff].index)
        print('valid kws', valid_kws)

        keywords = keywords.apply(lambda kws: [kw for kw in kws if kw in valid_kws])
        return keywords

    # all keywords for each article, except for quelle and date keywords
    keywords = preprocess_keywords(articles_with_keywords_and_probas.keywords)
    # prune infrequent keywords
    pruned_keywords = prune_keywords(keywords, 0.3)

    mlb = sklearn.preprocessing.MultiLabelBinarizer(sparse_output=True)
    Y = mlb.fit_transform(pruned_keywords)

    return Y, mlb.classes_


def inference_with_model(articles, save_path, Y_classes_save_path):
    """
    runs inference with the existing model on all of the articles flagged with INFERENCE.
    :return:
    """
    X = encode_features_of(articles)

    Y_classes = cPickle.load(open(Y_classes_save_path))
    clf = joblib.load(save_path)

    probas = clf.predict_proba(X)
    binarized = clf.predict(X)

    return probas, binarized, Y_classes


def train_model(articles_with_keywords_and_probas, clf_save_path, Y_classes_save_path):
    """
    model selection and evaluation with current set of training articles
    :return: None
    """
    # one row per keyword article group
    # articles_with_keywords = all_articles_by('TRAIN')

    print articles_with_keywords_and_probas.head()

    # encode data
    X = encode_features_of(articles_with_keywords_and_probas)
    Y, Y_classes = encode_labels_of(articles_with_keywords_and_probas)

    # model selection
    clf = cancer.ovr.classify_cancer(X, Y, Y_classes)

    # persistence
    joblib.dump(clf, clf_save_path)
    cPickle.dump(Y_classes, open(Y_classes_save_path, 'w'))
