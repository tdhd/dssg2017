import json # need double quotes in json POST bodies
from django.http import JsonResponse, HttpResponse
from cancer.models import RISArticle, RISArticleKeyword, all_articles_by, insert_with_keywords
import cancer.model_api.model
import django.conf
import cancer.persistence.models
import scipy.special
import numpy as np


def train(request):
    """
    Receives a POST-request with the following body:
        {
          'articles': [
            {
              'title': 'carc',
              'abstract': '...',
              'keywords': ['a', 'b', ...]
            },
            ...
          ]
        }

    and stores the data in disk-backed storage.
    Additionally, calling this endpoint will train the model on all of the training data uploaded.

    :param request: HTTP-request carrying all RIS article information.
    :return: http-status-code 200 in case of success.
    """
    request_body = json.loads(request.body)

    # add 1.0 as keyword_probabilities
    for article in request_body['articles']:
        article_keywords = article['keywords']
        del article['keywords']
        # keyword, proba, annotator
        kws = [(kw, 1.0, 'TRAIN') for kw in article_keywords]
        article['keywords'] = kws

    persistence = cancer.persistence.models.PandasPersistence(
        django.conf.settings.TRAIN_ARTICLES_PATH
    )
    persistence.save_batch(request_body['articles'])
    articles_with_keywords_and_probas = persistence.load_data()

    # retrain model and save to disk
    cancer.model_api.model.train_model(
        articles_with_keywords_and_probas,
        django.conf.settings.MODEL_PATH,
        django.conf.settings.LABEL_CODES_PATH
    )

    return HttpResponse(status=200)


def inference(request):
    """
    Receives a POST-request with the following body:
        {
          'articles': [
            {
              'title': 'carc',
              'abstract': '...',
            },
            ...
          ]
        }

    and runs multi-label inference on it.

    The results of the inference are stored as keywords.

    :param request: HTTP-request carrying all RIS articles (without keywords)
    :return: json response with all of the predicted labels from the uploaded RIS articles.
    """
    import cancer.model_api.model

    request_body = json.loads(request.body)

    persistence = cancer.persistence.models.PandasPersistence(
        django.conf.settings.INFERENCE_ARTICLES_PATH
    )
    for article in request_body['articles']:
        article['keywords'] = []
    persistence.save_batch(request_body['articles'])

    articles = persistence.load_data()

    # inference with current model
    label_probas, labels_binarized, label_names = cancer.model_api.model.inference_with_model(
        articles,
        django.conf.settings.MODEL_PATH,
        django.conf.settings.LABEL_CODES_PATH
    )

    # threshold predicted labels
    for index in range(articles.shape[0]):
        article = articles.ix[index]
        idcs = np.where(label_probas[index, :] >= django.conf.settings.INFERENCE_LABEL_THRESHOLD)
        article['keywords'] = [
            (name, proba, 'INFERENCE')
            for name, proba in zip(label_names[idcs], label_probas[index, idcs][0])
        ]

    # update storage with inferred keywords
    persistence.update(articles)

    # template rendering
    all_inference_articles = []
    for index, article in articles.iterrows():

        entry = {
            'article_id': index,
            'title': article.title,
            'abstract': article.abstract,
            'keywords': [
                {
                    'keyword': keyword,
                    'probability': proba,
                    'distance_to_hyperplane': scipy.special.logit(float(proba))
                } for keyword, proba, _ in article.keywords
            ]
        }
        all_inference_articles.append(entry)

    # TODO: make active learning strategy part of POST-request
    # active learning component for sorting
    import cancer.active_learning.selection_strategies
    sorted_articles = cancer.active_learning.selection_strategies.SelectionStrategies.default(
        all_inference_articles
    )

    return JsonResponse(
        {
            'article_predictions': sorted_articles
        }
    )


def update_model_with_feedback():
    """
    moves inference documents with feedback to train corpus and updates the model including that data.
    """
    # TODO: move data from INFERENCE to TRAIN

    # rerun model selection and persist
    cancer.model_api.model.update_model()


def feedback(request):
    """
    Receives a POST-request with a single document-label feedback pair:
    {
        'article_id': 123,
        'keyword': '4-harn,pall',
        'vote': 'OK',
        'annotator_name': 'annotator user name'
    }

    'vote' is in the set('OK', 'NOT OK').

    depending on 'vote' the keyword is either added or removed from the articles keywords.
    """
    should_retrain = False
    if should_retrain:
        # TODO: since there are all keywords stored per default
        # TODO: how to find a good strategy for label selection going from inf->train? (thresholding?)
        update_model_with_feedback()

    request_body = json.loads(request.body)

    # TODO: duplicate keywords possible

    if request_body['vote'] == 'OK':
        article = RISArticle.objects.get(id=request_body['article_id'])
        print article
        keyword = RISArticleKeyword(
            ris_article=article,
            keyword=request_body['keyword'],
            keyword_probability=1.0,
            annotator_name=request_body['annotator_name']
        )
        keyword.save()
    else:
        article = RISArticle.objects.get(id=request_body['article_id'])
        keyword = RISArticleKeyword.objects.filter(
            ris_article=article,
            keyword=request_body['keyword'],
            keyword_probability=1.0,
            annotator_name=request_body['annotator_name']
        )
        keyword.delete()

    return HttpResponse(status=200)
