# dssg2017

the latest file with all required code multi label prediction is in multiabel_test.ipynb.


## Files
* [truncate classifications](truncate_classifications.py), used for removing noisy or underrepresented labels from the dataset
* [top_level_labels](top_level_labels.py) to extract top level label lists from the CSV strings
* [multilabel_cancer_classification](multilabel_cancer_classification.py) pipeline for multi label classifications
* [multiabel_test](multiabel_test.ipynb) interactive pipeline for multi label classifications, including truncated classifications

The module [cleaning_classification_labels](cleaning_classification_labels) implements a cleaning pipeline for classifications which should be applied to rectify the labels a bit.

Also there is a notebook which we added in the beginning of the hack, [features.ipynb](features.ipynb), looking at different attributes of the features and also doing initial classification on `useful` label. 
