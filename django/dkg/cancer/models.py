# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models

# TODO: fully normalize schemas, especially to store keywords.


class TrainingArticle(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    title = models.TextField()
    abstract = models.TextField()
    authors = models.TextField()  # TSV
    keywords = models.TextField()  # TSV
    annotater_user_name = models.TextField()


class InferenceArticle(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    title = models.TextField()
    abstract = models.TextField()
    authors = models.TextField()  # TSV
    keywords = models.TextField()  # TSV
    annotater_user_name = models.TextField()
