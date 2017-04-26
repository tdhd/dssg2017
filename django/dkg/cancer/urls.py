from django.conf.urls import url

from . import views

urlpatterns = [
    url(r'^$', views.index, name='index'),
    url(r'model', views.model, name='model'),
    # url(r'upload_pred', views.upload_pred, name='upload_pred'),
]
