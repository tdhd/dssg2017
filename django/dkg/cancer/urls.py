from django.conf.urls import url

from . import views
from . import json_api

urlpatterns = [
    url(r'^$', views.index, name='index'),
    url(r'model', views.model, name='model'),

    url(r'json/train', json_api.train, name='train'),
    url(r'json/inference', json_api.inference, name='inference'),
    url(r'json/feedback_batch', json_api.feedback_batch, name='feedback_batch'),
    url(r'json/update_model', json_api.update_model, name='update_model'),

    url(r'download_test_ris', views.download_test_ris, name='download_test_ris'),
]
