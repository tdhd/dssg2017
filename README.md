# Data Science for Social Good 2017

March 2017 DSSG hack for the Deutsche Krebsgesellschaft (DKG).

We aim at building a multi-label model that is able to predict the labels for a given RIS article,
that includes features like:

* abstract
* title
* authors
* ...


## Webservice

We've built a django [webservice](django) that allows the DKG to interact with our model via RIS file uploads.

The service has the following features at the moment:

* upload training RIS file - triggers model selection on given data.
* upload test RIS file - produces keyword predictions for the given articles.

On each of the predictions, a user can give either positive or negative feedback, e.g. add another label
or remove a predicted label respectively.

### Active learning

In order to improve the mult-label model, the service is able to receive the feedback of a user.

We've implemented different strategies of prioritization for this active learning setting. See [this](http://burrsettles.com/pub/settles.activelearning.pdf)
article for a survey of active learning.

# Files from the hack weekend

* [truncate classifications](truncate_classifications.py), used for removing noisy or underrepresented labels from the dataset
* [top_level_labels](top_level_labels.py) to extract top level label lists from the CSV strings
* [multilabel_cancer_classification](multilabel_cancer_classification.py) pipeline for multi label classifications
* [multiabel_test](multiabel_test.ipynb) interactive pipeline for multi label classifications, including truncated classifications

The module [cleaning_classification_labels](cleaning_classification_labels) implements a cleaning pipeline for classifications which should be applied to rectify the labels a bit.

Also there is a notebook which we added in the beginning of the hack, [features.ipynb](features.ipynb), looking at different attributes of the features and also doing initial classification on `useful` label. 
