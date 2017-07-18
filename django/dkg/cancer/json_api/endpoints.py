import json # need double quotes in json POST bodies
from django.http import JsonResponse, HttpResponse
from cancer.models import RISArticle, RISArticleKeyword


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

    :param request: HTTP-request carrying all RIS article information.
    :return: http-status-code 200 in case of success.
    """
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
                annotator_name="json_train_upload_groundtruth"
            ) for kw in article['keywords']
        ]

        for keyword in keywords:
            keyword.save()

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

        :param article:
            article is a dictionary where the keys are the feature names, e.g.:
            {
                'title': '...',
                'abstract': '...
            }
        :return: list of (article-id, predicted clear-text labels).
        """
        return ['harn,1-karz', 'krk,4-med-adj']

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

        predicted_labels = label_predictions_for(article)
        # save all keywords for that article
        keywords = [
            RISArticleKeyword(ris_article=db_article, keyword=kw, annotator_name="scikit-model-1.0")
            for kw in label_predictions_for(article)
        ]

        for keyword in keywords:
            keyword.save()

            article_predictions.append(
            {
                'article_id': db_article.id,
                'predicted_keywords': predicted_labels
            }
        )

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
    # TODO: move data
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
