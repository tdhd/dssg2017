import json # need double quotes in json POST bodies
from django.http import JsonResponse, HttpResponse
import cancer.model_api.model
import django.conf
import cancer.persistence.models
import scipy.special
import numpy as np
import cancer.active_learning.selection_strategies
import cancer.model_api.model


def train(request):
    """
    Receives a POST-request with the following body:
        {
          'articles': [
            {
              'T1': 'carc',
              'N2': '...',
              'KW': ['a', 'b', ...],
              ... # more RIS keys https://en.wikipedia.org/wiki/RIS_(file_format)#Tags
            },
            ...
          ]
        }

    and stores the data in disk-backed storage.
    Additionally, calling this endpoint will train the model on all of the training data uploaded.

    :param request: HTTP-request carrying all RIS article information.
    :return: http-status-code 200 in case of success.
    """

    # TODO: make label pruning percentage part of request

    request_body = json.loads(request.body)

    # add 1.0 as keyword_probabilities
    for article in request_body['articles']:
        article_keywords = article['KW']
        del article['KW']
        # keyword, proba, annotator
        kws = [(kw, 1.0, 'TRAIN') for kw in article_keywords]
        article['KW'] = kws

    train_filename = cancer.persistence.models.persistence_filename(django.conf.settings.TRAIN_ARTICLES_PATTERN)
    persistence = cancer.persistence.models.PandasPersistence(
        train_filename
    )

    persistence.save_batch(request_body['articles'])
    articles_with_keywords_and_probas = persistence.load_data()

    # retrain model and save to disk
    cancer.model_api.model.train_model(
        articles_with_keywords_and_probas,
        django.conf.settings.MODEL_PATH,
        django.conf.settings.LABEL_CODES_PATH,
        django.conf.settings.FEATURE_ENCODER_PATH
    )

    return HttpResponse(status=200)


def inference(request):
    """
    Receives a POST-request with the following body:
        {
          'articles': [
            {
              'T1': 'carc',
              'N2': '...',
              ... # more key value pairs as in https://en.wikipedia.org/wiki/RIS_(file_format)#Tags
            },
            ...
          ]
        }

    and runs multi-label inference on it.

    The results of the inference are stored as keywords.

    :param request: HTTP-request carrying all RIS articles (without keywords)
    :return: json response with all of the predicted labels from the uploaded RIS articles.
    """
    def filename_with_prioritization_strategy(filename):
        tokens = filename.split(".")
        tokens[0] += '_sorting_prio_{}'.format('uncertainty_sampling' if use_active_learning_prio else 'random')
        return '.'.join(tokens)

    request_body = json.loads(request.body)

    use_active_learning_prio = django.conf.settings.ACTIVE_LEARNING_PRIO_PROBA < np.random.rand()
    print('Use active learning prio?', use_active_learning_prio)

    inference_filename = cancer.persistence.models.persistence_filename(django.conf.settings.INFERENCE_ARTICLES_PATTERN)
    inference_filename = filename_with_prioritization_strategy(inference_filename)

    persistence = cancer.persistence.models.PandasPersistence(
        inference_filename
    )
    for article in request_body['articles']:
        article['KW'] = []
    persistence.save_batch(request_body['articles'])

    articles = persistence.load_data()

    # inference with current model
    label_probas, labels_binarized, label_names = cancer.model_api.model.inference_with_model(
        articles,
        django.conf.settings.MODEL_PATH,
        django.conf.settings.LABEL_CODES_PATH,
        django.conf.settings.FEATURE_ENCODER_PATH
    )

    # threshold predicted labels
    for index in range(articles.shape[0]):
        article = articles.ix[index]
        idcs = np.where(label_probas[index, :] >= django.conf.settings.INFERENCE_LABEL_THRESHOLD)
        # idcs = np.where(labels_binarized[index, :].toarray().reshape(-1) == 1.0)
        article['KW'] = [
            (name, proba, 'INFERENCE')
            for name, proba in zip(label_names[idcs], label_probas[index, idcs][0])
        ]

    # update storage with inferred keywords
    # FIXME: disabled, only receive keyword predictions from both model and human annotator
    # persistence.update(articles)

    # template rendering
    all_inference_articles = []
    for index, article in articles.iterrows():

        entry = {
            'article_id': index,
            'title': article.T1,
            'abstract': article.N2,
            'keywords': [
                {
                    'keyword': keyword,
                    'probability': proba,
                    'distance_to_hyperplane': scipy.special.logit(float(proba))
                } for keyword, proba, _ in article.KW
            ]
        }
        all_inference_articles.append(entry)

    if use_active_learning_prio:
        sorted_articles = cancer.active_learning.selection_strategies.SelectionStrategies.default(
            all_inference_articles
        )
    else:
        from random import shuffle
        sorted_articles = all_inference_articles
        shuffle(sorted_articles)

    return JsonResponse(
        {
            'article_predictions': sorted_articles
        }
    )


# def update_model(request):
#     """
#     moves inference documents with feedback to train corpus and updates the model including that data.
#     """
#     import pandas as pd
#     train_persistence = cancer.persistence.models.PandasPersistence(
#         django.conf.settings.TRAIN_ARTICLES_PATH
#     )
#     train = train_persistence.load_data()
#     inference = cancer.persistence.models.PandasPersistence(
#         django.conf.settings.INFERENCE_ARTICLES_PATH
#     ).load_data()
#
#     merged = pd.concat((train, inference))
#
#     train_persistence.update(merged)
#
#     # retrain model and save to disk
#     cancer.model_api.model.train_model(
#         merged,
#         django.conf.settings.MODEL_PATH,
#         django.conf.settings.LABEL_CODES_PATH,
#         django.conf.settings.FEATURE_ENCODER_PATH
#     )
#
#     return HttpResponse(status=200)
#

# def feedback(request):
#     """
#     Receives a POST-request with a single document-label feedback pair:
#     {
#         'article_id': 123,
#         'keyword': '4-harn,pall',
#         'vote': 'OK',
#         'annotator_name': 'annotator user name'
#     }
#
#     'vote' is in the set('OK', 'NOT OK').
#
#     depending on 'vote' the keyword is either added or removed from the articles keywords.
#     """
#     def update_feedback_count():
#         import os.path
#         if os.path.isfile(django.conf.settings.FEEDBACK_COUNTER_PATH):
#             with open(django.conf.settings.FEEDBACK_COUNTER_PATH, 'r') as f:
#                 feedback_count = int(f.readline())
#             with open(django.conf.settings.FEEDBACK_COUNTER_PATH, 'w+') as f:
#                 f.write(str(feedback_count + 1))
#         else:
#             with open(django.conf.settings.FEEDBACK_COUNTER_PATH, 'w+') as f:
#                 f.write(str(1))
#
#     def n_feedbacks_outstanding():
#         with open(django.conf.settings.FEEDBACK_COUNTER_PATH, 'r') as f:
#             return int(f.readline())
#
#     def reset_feedback_count():
#         with open(django.conf.settings.FEEDBACK_COUNTER_PATH, 'w+') as f:
#             pass
#
#     update_feedback_count()
#
#     should_retrain = n_feedbacks_outstanding() > 10
#     if should_retrain:
#         reset_feedback_count()
#         update_model_with_feedback()
#
#     request_body = json.loads(request.body)
#
#     persistence = cancer.persistence.models.PandasPersistence(
#         django.conf.settings.INFERENCE_ARTICLES_PATH
#     )
#     inference_articles = persistence.load_data()
#
#     article = inference_articles.ix[np.int64(request_body['article_id'])]
#
#     if request_body['vote'] == 'OK':
#         # add feedback keyword to article keywords
#         article.KW = article.KW + [(request_body['keyword'], 1.0, request_body['annotator_name'])]
#     else:
#         # remove keyword
#         article.KW = [kw for kw in article.KW if kw[0] != request_body['keyword']]
#
#     persistence.update(inference_articles)
#
#     return HttpResponse(status=200)


def feedback_batch(request):
    """
    Receives a POST-request with all keywords for a single article:
    {
        'article_id': 123,
        'keywords': ['4-harn,pall', '1-med,car'],
        'annotator_name': 'annotator user name'
    }
    """
    request_body = json.loads(request.body)

    current_inference_filename = cancer.persistence.models.latest_persistence_filename(django.conf.settings.INFERENCE_ARTICLES_PATTERN)
    print 'updating inference file', current_inference_filename
    persistence = cancer.persistence.models.PandasPersistence(
        current_inference_filename
    )
    inference_articles = persistence.load_data()

    article = inference_articles.ix[np.int64(request_body['article_id'])]
    # replace all of the previous keywords
    article['KW'] = [(keyword.strip(), 1.0, request_body['annotator_name']) for keyword in request_body['keywords']]

    persistence.update(inference_articles)

    return HttpResponse(status=200)
