from django.conf.urls import url

from . import views
from . import json_api

urlpatterns = [
    url(r'^$', views.index, name='index'),
    url(r'model', views.model, name='model'),

    url(r'json/train', json_api.train, name='train'),
    url(r'json/inference', json_api.inference, name='inference'),
    url(r'json/feedback', json_api.feedback, name='feedback'),

    url(r'download_test_ris', views.download_test_ris, name='download_test_ris'),
]
