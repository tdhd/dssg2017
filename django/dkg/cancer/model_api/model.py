import sklearn.feature_extraction
import sklearn.preprocessing
import scipy.sparse
import cancer.ovr
import cancer.models
from sklearn.externals import joblib
import cPickle
import django.conf


def encode_features_of(articles_with_keywords_and_probas):
    """
    returns scipy.sparse matrix from model articles features.

    :param articles_with_keywords_and_probas: pandas.DataFrame.
    :return: sparse design matrix.
    """
    abstracts = sklearn.feature_extraction.text.HashingVectorizer(n_features=2**15)
    X_N2 = abstracts.fit_transform(articles_with_keywords_and_probas.N2)

    titles = sklearn.feature_extraction.text.HashingVectorizer(n_features=2**8)
    X_T1 = titles.fit_transform(articles_with_keywords_and_probas.T1)

    JA = sklearn.feature_extraction.text.HashingVectorizer(n_features=2**8)
    X_JA = JA.fit_transform(articles_with_keywords_and_probas.JA)

    JF = sklearn.feature_extraction.text.HashingVectorizer(n_features=2**8)
    X_JF = JF.fit_transform(articles_with_keywords_and_probas.JF)

    PB = sklearn.feature_extraction.text.HashingVectorizer(n_features=2**8)
    X_PB = PB.fit_transform(articles_with_keywords_and_probas.PB)

    X = scipy.sparse.hstack((X_T1, X_N2, X_JA, X_JF, X_PB))

    print 'encoded features shape {}'.format(X.shape)

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

        :param keywords: pandas.Series of lists with article keywords.
        :param p: cumsum relative keyword occurrence cutoff.
        :return:
        """
        import itertools
        import pandas as pd

        kws = pd.DataFrame({'kw': list(itertools.chain(*keywords))})
        print 'kws head', kws.head()
        csum_value_counts = kws.kw.value_counts(normalize=True).cumsum()
        cutoff = (csum_value_counts - p).abs().argmin()
        print csum_value_counts, cutoff
        print csum_value_counts.loc[:cutoff]
        valid_kws = list(csum_value_counts.loc[:cutoff].index.values)
        print valid_kws
        pruned_keywords = keywords.apply(lambda kws: [kw for kw in kws if kw in valid_kws])
        return pruned_keywords

    # all keywords for each article, except for quelle and date keywords
    keywords = preprocess_keywords(articles_with_keywords_and_probas['KW'])
    # prune infrequent keywords
    pruned_keywords = prune_keywords(
        keywords,
        django.conf.settings.LABEL_PRUNING_VALUE_COUNTS_THRESHOLD
    )

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
