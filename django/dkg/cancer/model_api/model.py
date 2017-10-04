import cPickle

import cancer.models
import cancer.ovr
import django.conf
import django.conf
import scipy.sparse
import sklearn.feature_extraction
import sklearn.preprocessing
from cancer.model_api.pipelining import ItemSelector
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline, FeatureUnion
import numpy as np


class LabelBinarizerPipelineFriendly(sklearn.preprocessing.LabelBinarizer):
    """
    quick hack to allow for labelbinarizer in pipeline
    cf. https://github.com/scikit-learn/scikit-learn/issues/3112
    """

    def fit(self, X, y=None):
        """this would allow us to fit the model based on the X input."""
        super(LabelBinarizerPipelineFriendly, self).fit(X)

    def transform(self, X, y=None):
        return super(LabelBinarizerPipelineFriendly, self).transform(X)

    def fit_transform(self, X, y=None):
        return super(LabelBinarizerPipelineFriendly, self).fit(X).transform(X)


def encoder_pipeline_from(articles_with_keywords_and_probas):
    """
    from pandas.DataFrame to Pipeline with FeatureUnion.

    :param articles_with_keywords_and_probas:
    :return: Pipeline with FeatureUnion to be able to .transform(df) to yield X matrix.
    """
    pipeline = Pipeline([
        ('union', FeatureUnion(
            transformer_list=[

                ('abstract', Pipeline([
                    ('selector', ItemSelector(key='N2')),
                    ('hasher', sklearn.feature_extraction.text.HashingVectorizer(n_features=2**12)),
                ])),

                ('title', Pipeline([
                    ('selector', ItemSelector(key='T1')),
                    ('hasher', sklearn.feature_extraction.text.HashingVectorizer(n_features=2**8))
                ])),

                ('journal-abbrev', Pipeline([
                    ('selector', ItemSelector(key='JA')),
                    ('hasher', sklearn.feature_extraction.text.HashingVectorizer(n_features=2**8))
                ])),

                ('journal-full', Pipeline([
                    ('selector', ItemSelector(key='JF')),
                    ('hasher', sklearn.feature_extraction.text.HashingVectorizer(n_features=2**8))
                ])),

                ('publisher', Pipeline([
                    ('selector', ItemSelector(key='PB')),
                    ('hasher', sklearn.feature_extraction.text.HashingVectorizer(n_features=2**8))
                ])),
                ('year', Pipeline([
                    ('selector', ItemSelector(key='Y1')),
                    ('hasher', sklearn.feature_extraction.text.HashingVectorizer(n_features=2**8))
                ])),

                # todo: probably should use pruning of publishers and years, too many distinct values
                # ('publisher_lb', Pipeline([
                #     ('selector', ItemSelector(key='PB')),
                #     ('lb', LabelBinarizerPipelineFriendly(sparse_output=True))
                # ])),
                # ('year_lb', Pipeline([
                #     ('selector', ItemSelector(key='Y1')),
                #     ('lb', LabelBinarizerPipelineFriendly(sparse_output=True))
                # ])),
            ]
        ))
    ])

    pipeline.fit(articles_with_keywords_and_probas)
    return pipeline


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


def inference_with_model(articles, save_path, Y_classes_save_path, feature_encoder_path):
    """
    runs inference with the existing model on all of the articles flagged with INFERENCE.
    :return:
    """
    encoder_pipeline = joblib.load(feature_encoder_path)
    X = encoder_pipeline.transform(articles)

    print X.shape

    Y_classes = cPickle.load(open(Y_classes_save_path))
    clf = joblib.load(save_path)

    probas = clf.predict_proba(X)
    binarized = clf.predict(X)

    return probas, binarized, Y_classes


def train_model(articles_with_keywords_and_probas, clf_save_path, Y_classes_save_path, feature_encoder_path):
    """
    model selection and evaluation with current set of training articles
    :return: None
    """
    # encode data
    encoder_pipeline = encoder_pipeline_from(articles_with_keywords_and_probas)
    X = encoder_pipeline.transform(articles_with_keywords_and_probas)
    Y, Y_classes = encode_labels_of(articles_with_keywords_and_probas)

    label_indicator = np.where(Y.sum(axis=1) > 0)[0]
    # only use samples which at least have one keyword
    X = X[label_indicator, :]
    Y = Y[label_indicator, :]

    # model selection
    clf = cancer.ovr.classify_cancer(X, Y, Y_classes)

    # persistence
    joblib.dump(clf, clf_save_path)
    joblib.dump(encoder_pipeline, feature_encoder_path)
    cPickle.dump(Y_classes, open(Y_classes_save_path, 'w'))
