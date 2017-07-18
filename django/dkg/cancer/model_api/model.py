def train_articles_adapter(model_articles):
    """
    given model training articles, returns compatible format for scikit pipeline.

    :param model_articles: n-element list of RISArticle
    :return: X, Y, clear text repr of Y columns
    """
    import sklearn.feature_extraction
    import sklearn.preprocessing
    import scipy.sparse

    def preprocess_keywords(keywords):
        """
        Lower-case keyword tokens and remove dates and search sources keywords.

        :param keywords: list of string keywords
        :return: cleaned list of string keywords
        """
        keywords_lowered = map(lambda kw: kw.lower(), keywords)
        kwt = filter(lambda kw: not kw.startswith('quelle,') and not kw.startswith('20'), keywords_lowered)
        return kwt

    abstracts = sklearn.feature_extraction.text.HashingVectorizer(n_features=2**15)
    X_abstracts = abstracts.fit_transform([article.abstract for article in model_articles])

    titles = sklearn.feature_extraction.text.HashingVectorizer(n_features=2**8)
    X_titles = titles.fit_transform([article.title for article in model_articles])

    mlb = sklearn.preprocessing.MultiLabelBinarizer(sparse_output=True)

    X = scipy.sparse.hstack((X_titles, X_abstracts))

    keywords = [preprocess_keywords(article.ts_keywords.split("\t")) for article in model_articles]
    Y = mlb.fit_transform(keywords)

    return X, Y, mlb.classes_


def inference_with_model():
    import cancer.models
    inference_articles = cancer.models.RISArticle.objects.filter(article_set='INFERENCE').all()
    # TODO: encode data

    # TODO: path to config
    clf = load_model('/tmp/test.pkl')
    probas = clf.predict_proba(X)

    return probas


def train_model():
    """
    model selection and evaluation with current set of training articles
    :return: scikit-learn classifier.
    """
    from cancer.models import all_training_articles_with_keywods
    import cancer.ovr
    # one row per keyword article group
    articles_with_keywords = all_training_articles_with_keywods()
    X, Y, Y_classes = train_articles_adapter(articles_with_keywords)
    print X.shape, Y.shape, Y_classes
    return cancer.ovr.classify_cancer(X, Y, Y_classes)


def update_model(save_path):
    """
    dumps trained model to save_path.

    :param save_path: path to where to save classifier to.
    """
    clf = train_model()
    from sklearn.externals import joblib
    joblib.dump(clf, save_path)


def load_model(save_path):
    from sklearn.externals import joblib
    return joblib.load(save_path)