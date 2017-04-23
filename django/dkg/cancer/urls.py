from django.conf.urls import url

from . import views

urlpatterns = [
    url(r'^$', views.index, name='index'),
    # url(r'upload_train', views.upload_train, name='upload_train'),
    # url(r'upload_pred', views.upload_pred, name='upload_pred'),
]
