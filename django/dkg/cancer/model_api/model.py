import numpy


def train_articles_adapter(model_articles):
    """
    given model training articles, returns compatible format for scikit pipeline.

    :param model_articles:
    :return: X, Y.
    """
    return numpy.zeros((100, 10)), numpy.zeros((100, 32))


def train_model():
    """
    model selection and evaluation with current set of training articles
    :return: scikit-learn classifier.
    """
    from cancer.models import TrainingArticle
    X, Y = train_articles_adapter(TrainingArticle.objects.all())
    # TODO: run model selection/evaluation here
    pass


def active_learning_feedback_to_training_corpus():
    """
    selects
    moves all of the inference articles with their (partially manually) annotated keywords to the training table.
    :return: None.
    """
    from cancer.models import InferenceArticle, TrainingArticle
    inference_articles = InferenceArticle.objects.all()
    # InferenceArticle.objects.delete()
    # save all inference_articles in TrainingArticle table
    pass

