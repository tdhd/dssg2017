import json # need double quotes in json POST bodies
from django.http import JsonResponse, HttpResponse
from cancer.models import RISArticle, RISArticleKeyword, all_articles_by, insert_with_keywords
import cancer.model_api.model
import django.conf
import scipy.special


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
    # delete all training data
    RISArticle.objects.filter(article_set='TRAIN').delete()

    request_body = json.loads(request.body)

    # add 1.0 as keyword_probabilities
    for article in request_body['articles']:
        article['keyword_probabilities'] = [1.0 for _ in range(len(article['keywords']))]

    insert_with_keywords(
        request_body['articles'],
        'TRAIN'
    )

    # retrain model and save to disk
    cancer.model_api.model.train_model(
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

    # delete all inference data
    # RISArticle.objects.filter(article_set='INFERENCE').delete()

    # request_body = json.loads(request.body)
    #
    # # run inference on the given articles, stores with inferred keywords in database
    # label_probas, label_names = cancer.model_api.model.inference_with_model(
    #     request_body['articles'],
    #     django.conf.settings.MODEL_PATH,
    #     django.conf.settings.LABEL_CODES_PATH
    # )
    #
    # insert_with_keywords(
    #     request_body['articles'],
    #     'INFERENCE'
    # )
    #
    # # save predicted keywords with their associated probability
    # for index in range(label_probas.shape[0]):
    #     for keyword_probability, keyword in zip(label_probas[index, :], label_names):
    #         kw = RISArticleKeyword(
    #             ris_article=db_articles[index],
    #             keyword=keyword,
    #             keyword_probability=keyword_probability,
    #             annotator_name="scikit-model-1.0"
    #         )
    #         kw.save()
    #
    # # select all of the inference articles with keywords
    # db_articles_with_keywords = all_articles_by('INFERENCE')
    # all_inference_articles = []
    # for article in db_articles_with_keywords:
    #
    #     entry = {
    #         'article_id': article.id,
    #         'title': article.title,
    #         'abstract': article.abstract
    #     }
    #
    #     entry_predictions = []
    #     for keyword, keyword_probability in zip(article.ts_keywords.split("\t"), article.ts_keyword_probabilities.split("\t")):
    #         distance_to_hyperplane = scipy.special.logit(float(keyword_probability))
    #
    #         entry_predictions.append(
    #             {
    #                 'keyword': keyword,
    #                 'probability': keyword_probability,
    #                 'distance_to_hyperplane': distance_to_hyperplane
    #             }
    #         )
    #
    #     entry['keywords'] = entry_predictions
    #     all_inference_articles.append(entry)
    #
    # # TODO: make active learning strategy part of POST-request
    # # active learning component for sorting
    # import cancer.active_learning.selection_strategies
    # sorted_articles = cancer.active_learning.selection_strategies.SelectionStrategies.default(
    #     all_inference_articles
    # )
    #
    # return JsonResponse(
    #     {
    #         'article_predictions': sorted_articles
    #     }
    # )


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
