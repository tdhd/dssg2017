import json # need double quotes in json POST bodies
from django.http import JsonResponse, HttpResponse
from cancer.models import RISArticle, RISArticleKeyword


def train(request):
    """
    Receives a POST-request with the following body:
        {
          'articles': [
            {
              'title': 'carcinomia',
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
            is_training_set=True
        )
        # save article
        db_article.save()

        # save all keywords for that article
        keywords = [
            RISArticleKeyword(ris_article=db_article, keyword=kw, annotater_name="123") for kw in article['keywords']
        ]

        for keyword in keywords:
            keyword.save()

    return HttpResponse(status=200)


def inference(request):
    """
    Receives a POST-request with the following body: ?
    and runs multi-label inference on it.

    :param request: HTTP-request carrying all RIS articles (without keywords)
    :param persistence: persistence object.
    :return: json response with all of the predicted labels from the uploaded RIS articles.

    Response body sample: ?
    """
    request_body = json.loads(request.body)
    print request_body
    # store all parsed articles for inference
    article = InferenceArticle(
        title='some article',
        abstract='abstract of article',
        authors='a\tb',
        keywords='4,harn-prog\t1,harn-pall'
    )
    article.save()
    return JsonResponse({'foo': 'bar'})


def feedback(request):
    """
    Receives a POST-request with a single document-label feedback pair.

    :param request:
    :param persistence: persistence object.
    :return:
    """
    request_body = json.loads(request.body)
    print request_body

    should_retrain = len(InferenceArticle.objects.all()) > 100

    # update InferenceArticle instance
    article = InferenceArticle.objects.get(pk=123)
    # update articles keywords
    article.keywords = "123\t321"
    article.save()
    return JsonResponse({'foo': 'bar'})
