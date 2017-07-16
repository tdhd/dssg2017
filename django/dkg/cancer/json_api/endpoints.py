import json # need double quotes in json POST bodies
from django.http import JsonResponse
from cancer.models import TrainingArticle, InferenceArticle


def train(request):
    """
    Receives a POST-request with the following body: ?

    Stores the data only.

    :param request: HTTP-request carrying all RIS article information.
    :param persistence: persistence object.
    :return: possible HTTP-status codes: 200, 400.
    """
    request_body = json.loads(request.body)
    print request_body
    # store all parsed articles
    article = TrainingArticle(
        title='some article',
        abstract='abstract of article',
        authors='a\tb',
        keywords='4,harn-prog\t1,harn-pall'
    )
    article.save()

    return JsonResponse({'foo': 'bar'})


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
