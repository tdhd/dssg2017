import sklearn.feature_extraction
import sklearn.preprocessing
import scipy.sparse


def X_encoder(model_articles):
    """
    returns scipy.sparse matrix from model articles features.

    :param model_articles: n-element list of RISArticle.
    :return: sparse design matrix.
    """
    abstracts = sklearn.feature_extraction.text.HashingVectorizer(n_features=2**15)
    X_abstracts = abstracts.fit_transform([article.abstract for article in model_articles])

    titles = sklearn.feature_extraction.text.HashingVectorizer(n_features=2**8)
    X_titles = titles.fit_transform([article.title for article in model_articles])

    X = scipy.sparse.hstack((X_titles, X_abstracts))
    return X


def Y_encoder(model_articles):
    """
    returns scipy.sparse matrix from model article labels.
    :param model_articles:
    :return:
    """
    def preprocess_keywords(keywords):
        """
        Lower-case keyword tokens and remove dates and search sources keywords.

        :param keywords: list of string keywords
        :return: cleaned list of string keywords
        """
        keywords_lowered = map(lambda kw: kw.lower(), keywords)
        kwt = filter(lambda kw: not kw.startswith('quelle,') and not kw.startswith('20'), keywords_lowered)
        return kwt

    keywords = [preprocess_keywords(article.ts_keywords.split("\t")) for article in model_articles]
    mlb = sklearn.preprocessing.MultiLabelBinarizer(sparse_output=True)
    Y = mlb.fit_transform(keywords)
    return Y, mlb.classes_


def inference_with_model(save_path, Y_classes_save_path):
    """
    runs inference with the existing model on all of the articles flagged with INFERENCE.
    :return:
    """

    import cancer.models
    inference_articles = cancer.models.RISArticle.objects.filter(article_set='INFERENCE').all()

    X = X_encoder(inference_articles)

    from sklearn.externals import joblib
    clf = joblib.load(save_path)
    import cPickle
    Y_classes = cPickle.load(open(Y_classes_save_path))

    probas = clf.predict_proba(X)
    return probas, Y_classes


def train_model(clf_save_path, Y_classes_save_path):
    """
    model selection and evaluation with current set of training articles
    :return: scikit-learn classifier and clear text mapping of classes
    """
    from cancer.models import all_training_articles_with_keywods
    import cancer.ovr
    # one row per keyword article group
    articles_with_keywords = all_training_articles_with_keywods()

    X = X_encoder(articles_with_keywords)
    Y, Y_classes = Y_encoder(articles_with_keywords)

    print X.shape, Y.shape, Y_classes
    clf = cancer.ovr.classify_cancer(X, Y, Y_classes)

    from sklearn.externals import joblib
    joblib.dump(clf, clf_save_path)

    import cPickle
    cPickle.dump(Y_classes, open(Y_classes_save_path, 'w'))

    return clf
