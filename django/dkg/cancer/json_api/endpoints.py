import json # need double quotes in json POST bodies
from django.http import JsonResponse, HttpResponse
from cancer.models import RISArticle, RISArticleKeyword
import cancer.model_api.model


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

    # repopulate
    request_body = json.loads(request.body)
    for article in request_body['articles']:
        db_article = RISArticle(
            title=article['title'],
            abstract=article['abstract'],
            article_set='TRAIN'
        )
        # save article
        db_article.save()

        # save all keywords for that article
        keywords = [
            RISArticleKeyword(
                ris_article=db_article,
                keyword=kw,
                annotator_name="json_train_upload_ground_truth"
            ) for kw in article['keywords']
        ]

        for keyword in keywords:
            keyword.save()

    # retrain model and save to disk
    # TODO: path should come from config
    cancer.model_api.model.update_model('/tmp/test.pkl')

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

    def label_predictions_for(article):
        """
        TODO: implement with scikit-backed model.
        TODO: move this model_api.model.py.

        :param article:
            article is a dictionary where the keys are the feature names, e.g.:
            {
                'title': '...',
                'abstract': '...
            }
        :return: same as input dictionary but with added keys:
            {
                'labels': [
                    {'name': '4-harn,carc', 'dist-to-hyperplane': 0.1},
                    {'name': '4-harn,pall', 'dist-to-hyperplane': 0.2}
                ]
            }
        """
        article['labels'] = [
            {'name': '4-harn,carc', 'dist-to-hyperplane': 0.1},
            {'name': '4-harn,pall', 'dist-to-hyperplane': 0.2}
        ]
        return article

    # delete all inference data
    RISArticle.objects.filter(article_set='INFERENCE').delete()

    request_body = json.loads(request.body)
    article_predictions = []
    for article in request_body['articles']:
        db_article = RISArticle(
            title=article['title'],
            abstract=article['abstract'],
            article_set='INFERENCE'
        )
        # save article
        db_article.save()

        article_with_labels = label_predictions_for(article)

        # save all keywords for that article
        keywords = [
            RISArticleKeyword(ris_article=db_article, keyword=label['name'], annotator_name="scikit-model-1.0")
            for label in article_with_labels['labels']
        ]

        for keyword in keywords:
            keyword.save()

            article_predictions.append(
                article_with_labels
            )

    # TODO: active learning prio strategy here

    return JsonResponse(
        {
            'article_predictions': article_predictions
        }
    )


def update_model_with_feedback():
    """
    moves inference documents with feedback to train corpus and updates the model including that data.
    """
    import cancer.model_api
    # TODO: move data from INFERENCE to TRAIN
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
        update_model_with_feedback()

    request_body = json.loads(request.body)

    if request_body['vote'] == 'OK':
        article = RISArticle.objects.get(id=request_body['article_id'])
        print article
        keyword = RISArticleKeyword(
            ris_article=article,
            keyword=request_body['keyword'],
            annotator_name=request_body['annotator_name']
        )
        keyword.save()
    else:
        article = RISArticle.objects.get(id=request_body['article_id'])
        keyword = RISArticleKeyword.objects.filter(
            ris_article=article,
            keyword=request_body['keyword'],
            annotator_name=request_body['annotator_name']
        )
        keyword.delete()

    return HttpResponse(status=200)
