# log of model selection


## `classifications_list_to_cancer_paper_subs`

top 100 labels from above transformation in classifications, classify cancer output

```
In [1]: from multilabel_cancer_classification import *

In [2]: classify_cancer()
Reading data for label extraction
Vectorizing labels
Vectorized 451 labels
Reading data for feature extraction
Vectorizing title character ngrams
Vectorizing keywords
Vectorizing authors
Vectorizing abstracts
Extracted feature vectors with 98356 dimensions
Truncating to top 100 labels accounted to 1.00 (45885/45885) of data (max count: 10031, min count: 117)
Training classifier
Fitting 3 folds for each of 24 candidates, totalling 72 fits
[CV] estimator__alpha=1e-06, estimator__penalty=l1, estimator__n_iter=20 
[CV] estimator__alpha=1e-06, estimator__penalty=l1, estimator__n_iter=20 
[CV] estimator__alpha=1e-06, estimator__penalty=l1, estimator__n_iter=20 
[CV] estimator__alpha=1e-06, estimator__penalty=l2, estimator__n_iter=20 
[CV] estimator__alpha=1e-06, estimator__penalty=l2, estimator__n_iter=20 
[CV] estimator__alpha=1e-06, estimator__penalty=l2, estimator__n_iter=20 
[CV] estimator__alpha=1e-06, estimator__penalty=l1, estimator__n_iter=40 
[CV] estimator__alpha=1e-06, estimator__penalty=l1, estimator__n_iter=40 
[CV]  estimator__alpha=1e-06, estimator__penalty=l2, estimator__n_iter=20, score=0.448436, total= 4.2min
[CV] estimator__alpha=1e-06, estimator__penalty=l1, estimator__n_iter=40 
[CV]  estimator__alpha=1e-06, estimator__penalty=l2, estimator__n_iter=20, score=0.445362, total= 4.2min
[CV] estimator__alpha=1e-06, estimator__penalty=l2, estimator__n_iter=40 
[CV]  estimator__alpha=1e-06, estimator__penalty=l2, estimator__n_iter=20, score=0.430256, total= 4.2min
[CV] estimator__alpha=1e-06, estimator__penalty=l2, estimator__n_iter=40 
[CV]  estimator__alpha=1e-06, estimator__penalty=l1, estimator__n_iter=20, score=0.408155, total=10.9min
[CV] estimator__alpha=1e-06, estimator__penalty=l2, estimator__n_iter=40 
[CV]  estimator__alpha=1e-06, estimator__penalty=l1, estimator__n_iter=20, score=0.407111, total=10.9min
[CV] estimator__alpha=1e-06, estimator__penalty=l1, estimator__n_iter=50 
[CV]  estimator__alpha=1e-06, estimator__penalty=l1, estimator__n_iter=20, score=0.394443, total=10.9min
[CV] estimator__alpha=1e-06, estimator__penalty=l1, estimator__n_iter=50 
[CV]  estimator__alpha=1e-06, estimator__penalty=l2, estimator__n_iter=40, score=0.459492, total= 7.9min
[CV] estimator__alpha=1e-06, estimator__penalty=l1, estimator__n_iter=50 
[CV]  estimator__alpha=1e-06, estimator__penalty=l2, estimator__n_iter=40, score=0.459074, total= 8.0min
[CV] estimator__alpha=1e-06, estimator__penalty=l2, estimator__n_iter=50 
[CV]  estimator__alpha=1e-06, estimator__penalty=l2, estimator__n_iter=40, score=0.442775, total= 7.9min
[CV] estimator__alpha=1e-06, estimator__penalty=l2, estimator__n_iter=50 
[CV]  estimator__alpha=1e-06, estimator__penalty=l1, estimator__n_iter=40, score=0.404953, total=21.6min
[CV] estimator__alpha=1e-06, estimator__penalty=l2, estimator__n_iter=50 
[CV]  estimator__alpha=1e-06, estimator__penalty=l1, estimator__n_iter=40, score=0.406098, total=21.7min
[CV] estimator__alpha=1e-06, estimator__penalty=l1, estimator__n_iter=60 
[CV]  estimator__alpha=1e-06, estimator__penalty=l2, estimator__n_iter=50, score=0.460270, total= 9.9min
[CV] estimator__alpha=1e-06, estimator__penalty=l1, estimator__n_iter=60 
[CV]  estimator__alpha=1e-06, estimator__penalty=l1, estimator__n_iter=40, score=0.388542, total=21.2min
[CV] estimator__alpha=1e-06, estimator__penalty=l1, estimator__n_iter=60 
[CV]  estimator__alpha=1e-06, estimator__penalty=l2, estimator__n_iter=50, score=0.460180, total= 9.9min
[CV] estimator__alpha=1e-06, estimator__penalty=l2, estimator__n_iter=60 
[CV]  estimator__alpha=1e-06, estimator__penalty=l2, estimator__n_iter=50, score=0.444274, total=10.0min
[CV] estimator__alpha=1e-06, estimator__penalty=l2, estimator__n_iter=60 
[CV]  estimator__alpha=1e-06, estimator__penalty=l1, estimator__n_iter=50, score=0.405999, total=26.4min
[Parallel(n_jobs=-1)]: Done  16 tasks      | elapsed: 37.7min
[CV] estimator__alpha=1e-06, estimator__penalty=l2, estimator__n_iter=60 
[CV]  estimator__alpha=1e-06, estimator__penalty=l1, estimator__n_iter=50, score=0.400151, total=26.5min
[CV] estimator__alpha=1e-05, estimator__penalty=l1, estimator__n_iter=20 
[CV]  estimator__alpha=1e-06, estimator__penalty=l1, estimator__n_iter=50, score=0.392935, total=26.5min
[CV] estimator__alpha=1e-05, estimator__penalty=l1, estimator__n_iter=20 
[CV]  estimator__alpha=1e-06, estimator__penalty=l2, estimator__n_iter=60, score=0.460799, total=11.8min
[CV] estimator__alpha=1e-05, estimator__penalty=l1, estimator__n_iter=20 
[CV]  estimator__alpha=1e-06, estimator__penalty=l2, estimator__n_iter=60, score=0.462659, total=11.8min
[CV] estimator__alpha=1e-05, estimator__penalty=l2, estimator__n_iter=20 
[CV]  estimator__alpha=1e-05, estimator__penalty=l2, estimator__n_iter=20, score=0.461701, total= 4.1min
[CV] estimator__alpha=1e-05, estimator__penalty=l2, estimator__n_iter=20 
[CV]  estimator__alpha=1e-05, estimator__penalty=l1, estimator__n_iter=20, score=0.427910, total=10.6min
[CV] estimator__alpha=1e-05, estimator__penalty=l2, estimator__n_iter=20 
[CV]  estimator__alpha=1e-06, estimator__penalty=l2, estimator__n_iter=60, score=0.446314, total=11.8min
[CV] estimator__alpha=1e-05, estimator__penalty=l1, estimator__n_iter=40 
[CV]  estimator__alpha=1e-05, estimator__penalty=l1, estimator__n_iter=20, score=0.431611, total=10.5min
[CV] estimator__alpha=1e-05, estimator__penalty=l1, estimator__n_iter=40 
[CV]  estimator__alpha=1e-05, estimator__penalty=l1, estimator__n_iter=20, score=0.418869, total=10.7min
[CV] estimator__alpha=1e-05, estimator__penalty=l1, estimator__n_iter=40 
[CV]  estimator__alpha=1e-05, estimator__penalty=l2, estimator__n_iter=20, score=0.466128, total= 4.1min
[CV] estimator__alpha=1e-05, estimator__penalty=l2, estimator__n_iter=40 
[CV]  estimator__alpha=1e-05, estimator__penalty=l2, estimator__n_iter=20, score=0.446480, total= 4.1min
[CV] estimator__alpha=1e-05, estimator__penalty=l2, estimator__n_iter=40 
[CV]  estimator__alpha=1e-06, estimator__penalty=l1, estimator__n_iter=60, score=0.405471, total=31.7min
[CV] estimator__alpha=1e-05, estimator__penalty=l2, estimator__n_iter=40 
[CV]  estimator__alpha=1e-06, estimator__penalty=l1, estimator__n_iter=60, score=0.402706, total=31.5min
[CV] estimator__alpha=1e-05, estimator__penalty=l1, estimator__n_iter=50 
[CV]  estimator__alpha=1e-06, estimator__penalty=l1, estimator__n_iter=60, score=0.392466, total=31.6min
[CV] estimator__alpha=1e-05, estimator__penalty=l1, estimator__n_iter=50 
[CV]  estimator__alpha=1e-05, estimator__penalty=l2, estimator__n_iter=40, score=0.461381, total= 7.9min
[CV] estimator__alpha=1e-05, estimator__penalty=l1, estimator__n_iter=50 
[CV]  estimator__alpha=1e-05, estimator__penalty=l2, estimator__n_iter=40, score=0.467095, total= 7.9min
[CV] estimator__alpha=1e-05, estimator__penalty=l2, estimator__n_iter=50 
[CV]  estimator__alpha=1e-05, estimator__penalty=l2, estimator__n_iter=40, score=0.446979, total= 7.9min
[CV] estimator__alpha=1e-05, estimator__penalty=l2, estimator__n_iter=50 
[CV]  estimator__alpha=1e-05, estimator__penalty=l1, estimator__n_iter=40, score=0.429122, total=20.9min
[CV] estimator__alpha=1e-05, estimator__penalty=l2, estimator__n_iter=50 
[CV]  estimator__alpha=1e-05, estimator__penalty=l2, estimator__n_iter=50, score=0.461418, total= 9.6min
[CV] estimator__alpha=1e-05, estimator__penalty=l1, estimator__n_iter=60 
[CV]  estimator__alpha=1e-05, estimator__penalty=l1, estimator__n_iter=40, score=0.433642, total=20.9min
[CV] estimator__alpha=1e-05, estimator__penalty=l1, estimator__n_iter=60 
[CV]  estimator__alpha=1e-05, estimator__penalty=l2, estimator__n_iter=50, score=0.467147, total= 9.5min
[CV] estimator__alpha=1e-05, estimator__penalty=l1, estimator__n_iter=60 
[CV]  estimator__alpha=1e-05, estimator__penalty=l1, estimator__n_iter=40, score=0.421942, total=21.2min
[CV] estimator__alpha=1e-05, estimator__penalty=l2, estimator__n_iter=60 
[CV]  estimator__alpha=1e-05, estimator__penalty=l2, estimator__n_iter=50, score=0.446865, total= 9.8min
[CV] estimator__alpha=1e-05, estimator__penalty=l2, estimator__n_iter=60 
[CV]  estimator__alpha=1e-05, estimator__penalty=l1, estimator__n_iter=50, score=0.431400, total=26.2min
[CV] estimator__alpha=1e-05, estimator__penalty=l2, estimator__n_iter=60 
[CV]  estimator__alpha=1e-05, estimator__penalty=l1, estimator__n_iter=50, score=0.436072, total=26.0min
[CV] estimator__alpha=0.0001, estimator__penalty=l1, estimator__n_iter=20 
[CV]  estimator__alpha=1e-05, estimator__penalty=l2, estimator__n_iter=60, score=0.461656, total=11.2min
[CV] estimator__alpha=0.0001, estimator__penalty=l1, estimator__n_iter=20 
[CV]  estimator__alpha=1e-05, estimator__penalty=l1, estimator__n_iter=50, score=0.423040, total=26.1min
[CV] estimator__alpha=0.0001, estimator__penalty=l1, estimator__n_iter=20 
[CV]  estimator__alpha=1e-05, estimator__penalty=l2, estimator__n_iter=60, score=0.447191, total=11.3min
[CV] estimator__alpha=0.0001, estimator__penalty=l2, estimator__n_iter=20 
[CV]  estimator__alpha=1e-05, estimator__penalty=l2, estimator__n_iter=60, score=0.467318, total=11.4min
[CV] estimator__alpha=0.0001, estimator__penalty=l2, estimator__n_iter=20 
[CV]  estimator__alpha=0.0001, estimator__penalty=l1, estimator__n_iter=20, score=0.320940, total=10.3min
[CV] estimator__alpha=0.0001, estimator__penalty=l2, estimator__n_iter=20 
[CV]  estimator__alpha=0.0001, estimator__penalty=l1, estimator__n_iter=20, score=0.333632, total= 9.7min
[CV] estimator__alpha=0.0001, estimator__penalty=l1, estimator__n_iter=40 
[CV]  estimator__alpha=0.0001, estimator__penalty=l2, estimator__n_iter=20, score=0.386992, total= 3.6min
[CV] estimator__alpha=0.0001, estimator__penalty=l1, estimator__n_iter=40 
[CV]  estimator__alpha=0.0001, estimator__penalty=l2, estimator__n_iter=20, score=0.380050, total= 3.8min
[CV] estimator__alpha=0.0001, estimator__penalty=l1, estimator__n_iter=40 
[CV]  estimator__alpha=0.0001, estimator__penalty=l1, estimator__n_iter=20, score=0.318653, total=10.4min
[CV] estimator__alpha=0.0001, estimator__penalty=l2, estimator__n_iter=40 
[CV]  estimator__alpha=0.0001, estimator__penalty=l2, estimator__n_iter=20, score=0.367020, total= 3.9min
[CV] estimator__alpha=0.0001, estimator__penalty=l2, estimator__n_iter=40 
[CV]  estimator__alpha=1e-05, estimator__penalty=l1, estimator__n_iter=60, score=0.434311, total=31.0min
[CV] estimator__alpha=0.0001, estimator__penalty=l2, estimator__n_iter=40 
[CV]  estimator__alpha=1e-05, estimator__penalty=l1, estimator__n_iter=60, score=0.430662, total=31.2min
[CV] estimator__alpha=0.0001, estimator__penalty=l1, estimator__n_iter=50 
[CV]  estimator__alpha=1e-05, estimator__penalty=l1, estimator__n_iter=60, score=0.423833, total=31.3min
[CV] estimator__alpha=0.0001, estimator__penalty=l1, estimator__n_iter=50 
[CV]  estimator__alpha=0.0001, estimator__penalty=l2, estimator__n_iter=40, score=0.380130, total= 7.4min
[CV] estimator__alpha=0.0001, estimator__penalty=l1, estimator__n_iter=50 
[CV]  estimator__alpha=0.0001, estimator__penalty=l2, estimator__n_iter=40, score=0.387646, total= 7.5min
[CV] estimator__alpha=0.0001, estimator__penalty=l2, estimator__n_iter=50 
[CV]  estimator__alpha=0.0001, estimator__penalty=l2, estimator__n_iter=40, score=0.366711, total= 7.4min
[CV] estimator__alpha=0.0001, estimator__penalty=l2, estimator__n_iter=50 
[CV]  estimator__alpha=0.0001, estimator__penalty=l1, estimator__n_iter=40, score=0.322867, total=19.1min
[CV] estimator__alpha=0.0001, estimator__penalty=l2, estimator__n_iter=50 
[CV]  estimator__alpha=0.0001, estimator__penalty=l1, estimator__n_iter=40, score=0.335427, total=19.1min
[CV] estimator__alpha=0.0001, estimator__penalty=l1, estimator__n_iter=60 
[CV]  estimator__alpha=0.0001, estimator__penalty=l2, estimator__n_iter=50, score=0.380138, total= 9.3min
[CV] estimator__alpha=0.0001, estimator__penalty=l1, estimator__n_iter=60 
[CV]  estimator__alpha=0.0001, estimator__penalty=l1, estimator__n_iter=40, score=0.320076, total=20.3min
[CV] estimator__alpha=0.0001, estimator__penalty=l1, estimator__n_iter=60 
[CV]  estimator__alpha=0.0001, estimator__penalty=l2, estimator__n_iter=50, score=0.387653, total= 9.8min
[CV] estimator__alpha=0.0001, estimator__penalty=l2, estimator__n_iter=60 
[CV]  estimator__alpha=0.0001, estimator__penalty=l2, estimator__n_iter=50, score=0.366892, total= 9.3min
[CV] estimator__alpha=0.0001, estimator__penalty=l2, estimator__n_iter=60 
[CV]  estimator__alpha=0.0001, estimator__penalty=l1, estimator__n_iter=50, score=0.322196, total=25.1min
[CV] estimator__alpha=0.0001, estimator__penalty=l2, estimator__n_iter=60 
[CV]  estimator__alpha=0.0001, estimator__penalty=l1, estimator__n_iter=50, score=0.336652, total=28.6min
[CV]  estimator__alpha=0.0001, estimator__penalty=l2, estimator__n_iter=60, score=0.380387, total=13.6min
[CV]  estimator__alpha=0.0001, estimator__penalty=l1, estimator__n_iter=50, score=0.320688, total=28.9min
[CV]  estimator__alpha=0.0001, estimator__penalty=l2, estimator__n_iter=60, score=0.387773, total=13.2min
[CV]  estimator__alpha=0.0001, estimator__penalty=l2, estimator__n_iter=60, score=0.367146, total=11.0min
[CV]  estimator__alpha=0.0001, estimator__penalty=l1, estimator__n_iter=60, score=0.324020, total=27.6min
[CV]  estimator__alpha=0.0001, estimator__penalty=l1, estimator__n_iter=60, score=0.336969, total=27.6min
[CV]  estimator__alpha=0.0001, estimator__penalty=l1, estimator__n_iter=60, score=0.320625, total=27.2min
[Parallel(n_jobs=-1)]: Done  72 out of  72 | elapsed: 143.9min finished
Model with rank: 1
Mean validation score: 0.459 (std: 0.008)
Parameters: {'estimator__alpha': 1e-05, 'estimator__penalty': 'l2', 'estimator__n_iter': 60}

Model with rank: 2
Mean validation score: 0.458 (std: 0.008)
Parameters: {'estimator__alpha': 1e-05, 'estimator__penalty': 'l2', 'estimator__n_iter': 40}

Model with rank: 3
Mean validation score: 0.458 (std: 0.009)
Parameters: {'estimator__alpha': 1e-05, 'estimator__penalty': 'l2', 'estimator__n_iter': 50}

{'recall_score': [('mesotheliom', 0.94736842105263153), ('kopf', 0.8549107142857143), ('oesophagus', 0.85416666666666663), ('lunge', 0.85301837270341208), ('schilddruese', 0.84375), ('magen', 0.82804232804232802), ('zervix', 0.81749049429657794), ('pharynxlarynx', 0.81355932203389836), ('nsclc', 0.79699248120300747), ('harn', 0.74592833876221498), ('pankreas', 0.74294670846394983), ('leber', 0.73898305084745763), ('gist', 0.72093023255813948), ('endometrium', 0.7103825136612022), ('anus', 0.65306122448979587), ('hirn', 0.62780269058295968), ('4', 0.60072297075254688), ('1', 0.60019646365422397), ('niere', 0.56896551724137934), ('glioblastom', 0.54545454545454541), ('pall', 0.52692867540029109), ('hpv', 0.51282051282051277), ('med', 0.48853754940711464), ('prostata', 0.46727549467275492), ('schm', 0.45588235294117646), ('vulva', 0.44117647058823528), ('ovar', 0.40999999999999998), ('infekt', 0.40540540540540543), ('basis', 0.40059347181008903), ('galle', 0.39805825242718446), ('nm', 0.38461538461538464), ('melanom', 0.37037037037037035), ('bew', 0.37037037037037035), ('5', 0.36954503249767873), ('3', 0.34715025906735753), ('mikro', 0.33035714285714285), ('hoden', 0.32786885245901637), ('op', 0.30959302325581395), ('obstrukt', 0.30909090909090908), ('alk', 0.30769230769230771), ('karz', 0.30681818181818182), ('fat', 0.30434782608695654), ('progn', 0.28460038986354774), ('rauch', 0.28333333333333333), ('pet', 0.27777777777777779), ('kn', 0.25925925925925924), ('endo', 0.25), ('ue', 0.24390243902439024), ('net', 0.23076923076923078), ('marker', 0.22727272727272727), ('screen', 0.224), ('tu', 0.21621621621621623), ('sclc', 0.21052631578947367), ('mamma', 0.19881889763779528), ('diab', 0.1875), ('met', 0.18681318681318682), ('6', 0.17962466487935658), ('rch', 0.1761006289308176), ('gew', 0.16981132075471697), ('lymphome', 0.16666666666666666), ('krk', 0.16450216450216451), ('mrt', 0.15873015873015872), ('adj', 0.14981273408239701), ('gliom', 0.14634146341463414), ('lap', 0.14634146341463414), ('str', 0.14344262295081966), ('ern', 0.14141414141414141), ('alter', 0.13461538461538461), ('spaet', 0.13333333333333333), ('biops', 0.13186813186813187), ('insitu', 0.1276595744680851), ('leukaemien', 0.125), ('und', 0.12244897959183673), ('and', 0.12236286919831224), ('8', 0.12195121951219512), ('lq', 0.11206896551724138), ('po', 0.1111111111111111), ('mund', 0.10714285714285714), ('gen', 0.096153846153846159), ('erk', 0.095588235294117641), ('kam', 0.085365853658536592), ('neo', 0.071428571428571425), ('7', 0.066666666666666666), ('2', 0.065727699530516437), ('kur', 0.064814814814814811), ('sex', 0.057142857142857141), ('sono', 0.054054054054054057), ('ct', 0.052631578947368418), ('makro', 0.04878048780487805), ('ln', 0.04878048780487805), ('spez', 0.047872340425531915), ('haut', 0.045454545454545456), ('horm', 0.045454545454545456), ('arten', 0.044776119402985072), ('frueh', 0.034782608695652174), ('bio', 0.027397260273972601), ('rez', 0.026315789473684209), ('bb', 0.025000000000000001), ('nw', 0.020202020202020204), ('histo', 0.019230769230769232)], 'f1_score': [('schilddruese', 0.82442748091603046), ('magen', 0.8236842105263158), ('kopf', 0.81230116648992579), ('oesophagus', 0.81188118811881205), ('nsclc', 0.81122448979591832), ('lunge', 0.8099688473520249), ('mesotheliom', 0.80898876404494391), ('zervix', 0.7719928186714542), ('pankreas', 0.77073170731707308), ('pharynxlarynx', 0.75949367088607589), ('harn', 0.74836601307189543), ('leber', 0.7218543046357615), ('gist', 0.72093023255813948), ('endometrium', 0.70652173913043481), ('anus', 0.68817204301075263), ('hirn', 0.67796610169491511), ('glioblastom', 0.65934065934065922), ('4', 0.65146115466856735), ('1', 0.65069222577209795), ('niere', 0.62411347517730498), ('pall', 0.60232945091514145), ('hpv', 0.59701492537313428), ('med', 0.57515123313168914), ('schm', 0.55605381165919288), ('ovar', 0.52676659528907921), ('vulva', 0.52631578947368418), ('infekt', 0.51724137931034486), ('basis', 0.51461245235069886), ('nm', 0.51428571428571423), ('prostata', 0.50869925434962715), ('galle', 0.5), ('5', 0.46796002351557903), ('melanom', 0.46783625730994155), ('3', 0.4589041095890411), ('fat', 0.45161290322580649), ('obstrukt', 0.44155844155844154), ('mikro', 0.43786982248520717), ('karz', 0.43548387096774188), ('bew', 0.43478260869565216), ('hoden', 0.43010752688172044), ('op', 0.42642642642642642), ('rauch', 0.42499999999999999), ('progn', 0.40668523676880219), ('alk', 0.3902439024390244), ('kn', 0.3888888888888889), ('ue', 0.37037037037037035), ('endo', 0.36879432624113478), ('pet', 0.36697247706422015), ('screen', 0.35000000000000003), ('marker', 0.34722222222222221), ('net', 0.339622641509434), ('sclc', 0.33802816901408456), ('tu', 0.33566433566433568), ('diab', 0.29999999999999999), ('met', 0.29059829059829057), ('gew', 0.29032258064516131), ('rch', 0.28426395939086296), ('mamma', 0.28133704735376047), ('6', 0.26534653465346536), ('krk', 0.2567567567567568), ('lymphome', 0.25396825396825395), ('mrt', 0.25), ('gliom', 0.23529411764705882), ('ern', 0.23529411764705879), ('lap', 0.23300970873786406), ('alter', 0.22580645161290322), ('adj', 0.2247191011235955), ('str', 0.22364217252396165), ('insitu', 0.22222222222222221), ('biops', 0.22018348623853212), ('spaet', 0.2142857142857143), ('8', 0.2061855670103093), ('and', 0.20000000000000001), ('leukaemien', 0.20000000000000001), ('und', 0.19672131147540983), ('lq', 0.19402985074626866), ('po', 0.1818181818181818), ('mund', 0.17475728155339806), ('erk', 0.16993464052287582), ('gen', 0.15625), ('kam', 0.14893617021276598), ('neo', 0.12903225806451613), ('kur', 0.11965811965811966), ('7', 0.11904761904761904), ('2', 0.1166666666666667), ('sono', 0.097560975609756101), ('sex', 0.097560975609756087), ('ct', 0.097560975609756087), ('makro', 0.090909090909090898), ('ln', 0.090909090909090898), ('spez', 0.087804878048780496), ('horm', 0.085714285714285715), ('haut', 0.083333333333333343), ('arten', 0.080536912751677847), ('frueh', 0.066666666666666666), ('bio', 0.051948051948051945), ('rez', 0.050000000000000003), ('bb', 0.048780487804878057), ('nw', 0.039603960396039611), ('histo', 0.037735849056603779)], 'precision_score': [('gew', 1.0), ('bb', 1.0), ('histo', 1.0), ('nw', 1.0), ('fat', 0.875), ('insitu', 0.8571428571428571), ('sclc', 0.8571428571428571), ('rauch', 0.84999999999999998), ('glioblastom', 0.83333333333333337), ('nsclc', 0.82597402597402603), ('magen', 0.81937172774869105), ('schilddruese', 0.80597014925373134), ('pankreas', 0.80067567567567566), ('screen', 0.80000000000000004), ('frueh', 0.80000000000000004), ('kur', 0.77777777777777779), ('kn', 0.77777777777777779), ('nm', 0.77586206896551724), ('kopf', 0.77373737373737372), ('oesophagus', 0.77358490566037741), ('obstrukt', 0.77272727272727271), ('lunge', 0.77105575326215892), ('ue', 0.76923076923076927), ('erk', 0.76470588235294112), ('harn', 0.75081967213114753), ('horm', 0.75), ('diab', 0.75), ('tu', 0.75), ('karz', 0.75), ('rch', 0.73684210526315785), ('hirn', 0.73684210526315785), ('ovar', 0.73652694610778446), ('marker', 0.73529411764705888), ('zervix', 0.73129251700680276), ('anus', 0.72727272727272729), ('lq', 0.72222222222222221), ('gist', 0.72093023255813948), ('basis', 0.71936056838365892), ('hpv', 0.7142857142857143), ('infekt', 0.7142857142857143), ('schm', 0.71264367816091956), ('progn', 0.71219512195121948), ('pharynxlarynx', 0.71216617210682498), ('4', 0.71156091864538729), ('1', 0.71046511627906972), ('mesotheliom', 0.70588235294117652), ('leber', 0.70550161812297729), ('pall', 0.70291262135922328), ('endometrium', 0.70270270270270274), ('endo', 0.70270270270270274), ('ern', 0.69999999999999996), ('alter', 0.69999999999999996), ('med', 0.69909502262443435), ('niere', 0.69109947643979053), ('op', 0.68488745980707399), ('3', 0.6767676767676768), ('galle', 0.67213114754098358), ('ct', 0.66666666666666663), ('ln', 0.66666666666666663), ('biops', 0.66666666666666663), ('makro', 0.66666666666666663), ('neo', 0.66666666666666663), ('8', 0.66666666666666663), ('met', 0.65384615384615385), ('vulva', 0.65217391304347827), ('mikro', 0.64912280701754388), ('net', 0.6428571428571429), ('5', 0.63782051282051277), ('melanom', 0.63492063492063489), ('hoden', 0.625), ('gliom', 0.59999999999999998), ('mrt', 0.58823529411764708), ('krk', 0.58461538461538465), ('kam', 0.58333333333333337), ('lap', 0.5714285714285714), ('prostata', 0.55818181818181822), ('7', 0.55555555555555558), ('and', 0.54716981132075471), ('spaet', 0.54545454545454541), ('pet', 0.54054054054054057), ('alk', 0.53333333333333333), ('lymphome', 0.53333333333333333), ('spez', 0.52941176470588236), ('bew', 0.52631578947368418), ('2', 0.51851851851851849), ('6', 0.50757575757575757), ('str', 0.50724637681159424), ('po', 0.5), ('und', 0.5), ('haut', 0.5), ('sono', 0.5), ('rez', 0.5), ('leukaemien', 0.5), ('bio', 0.5), ('mamma', 0.48095238095238096), ('mund', 0.47368421052631576), ('adj', 0.449438202247191), ('gen', 0.41666666666666669), ('arten', 0.40000000000000002), ('sex', 0.33333333333333331)]}
Retraining on all data
Reading data for testing model
Reading data for feature extraction
Vectorizing title character ngrams
Vectorizing keywords
Vectorizing authors
Vectorizing abstracts
Extracted feature vectors with 98356 dimensions
Out[2]: 
({'f1_score': [('schilddruese', 0.82442748091603046),
   ('magen', 0.8236842105263158),
   ('kopf', 0.81230116648992579),
   ('oesophagus', 0.81188118811881205),
   ('nsclc', 0.81122448979591832),
   ('lunge', 0.8099688473520249),
   ('mesotheliom', 0.80898876404494391),
   ('zervix', 0.7719928186714542),
   ('pankreas', 0.77073170731707308),
   ('pharynxlarynx', 0.75949367088607589),
   ('harn', 0.74836601307189543),
   ('leber', 0.7218543046357615),
   ('gist', 0.72093023255813948),
   ('endometrium', 0.70652173913043481),
   ('anus', 0.68817204301075263),
   ('hirn', 0.67796610169491511),
   ('glioblastom', 0.65934065934065922),
   ('4', 0.65146115466856735),
   ('1', 0.65069222577209795),
   ('niere', 0.62411347517730498),
   ('pall', 0.60232945091514145),
   ('hpv', 0.59701492537313428),
   ('med', 0.57515123313168914),
   ('schm', 0.55605381165919288),
   ('ovar', 0.52676659528907921),
   ('vulva', 0.52631578947368418),
   ('infekt', 0.51724137931034486),
   ('basis', 0.51461245235069886),
   ('nm', 0.51428571428571423),
   ('prostata', 0.50869925434962715),
   ('galle', 0.5),
   ('5', 0.46796002351557903),
   ('melanom', 0.46783625730994155),
   ('3', 0.4589041095890411),
   ('fat', 0.45161290322580649),
   ('obstrukt', 0.44155844155844154),
   ('mikro', 0.43786982248520717),
   ('karz', 0.43548387096774188),
   ('bew', 0.43478260869565216),
   ('hoden', 0.43010752688172044),
   ('op', 0.42642642642642642),
   ('rauch', 0.42499999999999999),
   ('progn', 0.40668523676880219),
   ('alk', 0.3902439024390244),
   ('kn', 0.3888888888888889),
   ('ue', 0.37037037037037035),
   ('endo', 0.36879432624113478),
   ('pet', 0.36697247706422015),
   ('screen', 0.35000000000000003),
   ('marker', 0.34722222222222221),
   ('net', 0.339622641509434),
   ('sclc', 0.33802816901408456),
   ('tu', 0.33566433566433568),
   ('diab', 0.29999999999999999),
   ('met', 0.29059829059829057),
   ('gew', 0.29032258064516131),
   ('rch', 0.28426395939086296),
   ('mamma', 0.28133704735376047),
   ('6', 0.26534653465346536),
   ('krk', 0.2567567567567568),
   ('lymphome', 0.25396825396825395),
   ('mrt', 0.25),
   ('gliom', 0.23529411764705882),
   ('ern', 0.23529411764705879),
   ('lap', 0.23300970873786406),
   ('alter', 0.22580645161290322),
   ('adj', 0.2247191011235955),
   ('str', 0.22364217252396165),
   ('insitu', 0.22222222222222221),
   ('biops', 0.22018348623853212),
   ('spaet', 0.2142857142857143),
   ('8', 0.2061855670103093),
   ('and', 0.20000000000000001),
   ('leukaemien', 0.20000000000000001),
   ('und', 0.19672131147540983),
   ('lq', 0.19402985074626866),
   ('po', 0.1818181818181818),
   ('mund', 0.17475728155339806),
   ('erk', 0.16993464052287582),
   ('gen', 0.15625),
   ('kam', 0.14893617021276598),
   ('neo', 0.12903225806451613),
   ('kur', 0.11965811965811966),
   ('7', 0.11904761904761904),
   ('2', 0.1166666666666667),
   ('sono', 0.097560975609756101),
   ('sex', 0.097560975609756087),
   ('ct', 0.097560975609756087),
   ('makro', 0.090909090909090898),
   ('ln', 0.090909090909090898),
   ('spez', 0.087804878048780496),
   ('horm', 0.085714285714285715),
   ('haut', 0.083333333333333343),
   ('arten', 0.080536912751677847),
   ('frueh', 0.066666666666666666),
   ('bio', 0.051948051948051945),
   ('rez', 0.050000000000000003),
   ('bb', 0.048780487804878057),
   ('nw', 0.039603960396039611),
   ('histo', 0.037735849056603779)],
  'precision_score': [('gew', 1.0),
   ('bb', 1.0),
   ('histo', 1.0),
   ('nw', 1.0),
   ('fat', 0.875),
   ('insitu', 0.8571428571428571),
   ('sclc', 0.8571428571428571),
   ('rauch', 0.84999999999999998),
   ('glioblastom', 0.83333333333333337),
   ('nsclc', 0.82597402597402603),
   ('magen', 0.81937172774869105),
   ('schilddruese', 0.80597014925373134),
   ('pankreas', 0.80067567567567566),
   ('screen', 0.80000000000000004),
   ('frueh', 0.80000000000000004),
   ('kur', 0.77777777777777779),
   ('kn', 0.77777777777777779),
   ('nm', 0.77586206896551724),
   ('kopf', 0.77373737373737372),
   ('oesophagus', 0.77358490566037741),
   ('obstrukt', 0.77272727272727271),
   ('lunge', 0.77105575326215892),
   ('ue', 0.76923076923076927),
   ('erk', 0.76470588235294112),
   ('harn', 0.75081967213114753),
   ('horm', 0.75),
   ('diab', 0.75),
   ('tu', 0.75),
   ('karz', 0.75),
   ('rch', 0.73684210526315785),
   ('hirn', 0.73684210526315785),
   ('ovar', 0.73652694610778446),
   ('marker', 0.73529411764705888),
   ('zervix', 0.73129251700680276),
   ('anus', 0.72727272727272729),
   ('lq', 0.72222222222222221),
   ('gist', 0.72093023255813948),
   ('basis', 0.71936056838365892),
   ('hpv', 0.7142857142857143),
   ('infekt', 0.7142857142857143),
   ('schm', 0.71264367816091956),
   ('progn', 0.71219512195121948),
   ('pharynxlarynx', 0.71216617210682498),
   ('4', 0.71156091864538729),
   ('1', 0.71046511627906972),
   ('mesotheliom', 0.70588235294117652),
   ('leber', 0.70550161812297729),
   ('pall', 0.70291262135922328),
   ('endometrium', 0.70270270270270274),
   ('endo', 0.70270270270270274),
   ('ern', 0.69999999999999996),
   ('alter', 0.69999999999999996),
   ('med', 0.69909502262443435),
   ('niere', 0.69109947643979053),
   ('op', 0.68488745980707399),
   ('3', 0.6767676767676768),
   ('galle', 0.67213114754098358),
   ('ct', 0.66666666666666663),
   ('ln', 0.66666666666666663),
   ('biops', 0.66666666666666663),
   ('makro', 0.66666666666666663),
   ('neo', 0.66666666666666663),
   ('8', 0.66666666666666663),
   ('met', 0.65384615384615385),
   ('vulva', 0.65217391304347827),
   ('mikro', 0.64912280701754388),
   ('net', 0.6428571428571429),
   ('5', 0.63782051282051277),
   ('melanom', 0.63492063492063489),
   ('hoden', 0.625),
   ('gliom', 0.59999999999999998),
   ('mrt', 0.58823529411764708),
   ('krk', 0.58461538461538465),
   ('kam', 0.58333333333333337),
   ('lap', 0.5714285714285714),
   ('prostata', 0.55818181818181822),
   ('7', 0.55555555555555558),
   ('and', 0.54716981132075471),
   ('spaet', 0.54545454545454541),
   ('pet', 0.54054054054054057),
   ('alk', 0.53333333333333333),
   ('lymphome', 0.53333333333333333),
   ('spez', 0.52941176470588236),
   ('bew', 0.52631578947368418),
   ('2', 0.51851851851851849),
   ('6', 0.50757575757575757),
   ('str', 0.50724637681159424),
   ('po', 0.5),
   ('und', 0.5),
   ('haut', 0.5),
   ('sono', 0.5),
   ('rez', 0.5),
   ('leukaemien', 0.5),
   ('bio', 0.5),
   ('mamma', 0.48095238095238096),
   ('mund', 0.47368421052631576),
   ('adj', 0.449438202247191),
   ('gen', 0.41666666666666669),
   ('arten', 0.40000000000000002),
   ('sex', 0.33333333333333331)],
  'recall_score': [('mesotheliom', 0.94736842105263153),
   ('kopf', 0.8549107142857143),
   ('oesophagus', 0.85416666666666663),
   ('lunge', 0.85301837270341208),
   ('schilddruese', 0.84375),
   ('magen', 0.82804232804232802),
   ('zervix', 0.81749049429657794),
   ('pharynxlarynx', 0.81355932203389836),
   ('nsclc', 0.79699248120300747),
   ('harn', 0.74592833876221498),
   ('pankreas', 0.74294670846394983),
   ('leber', 0.73898305084745763),
   ('gist', 0.72093023255813948),
   ('endometrium', 0.7103825136612022),
   ('anus', 0.65306122448979587),
   ('hirn', 0.62780269058295968),
   ('4', 0.60072297075254688),
   ('1', 0.60019646365422397),
   ('niere', 0.56896551724137934),
   ('glioblastom', 0.54545454545454541),
   ('pall', 0.52692867540029109),
   ('hpv', 0.51282051282051277),
   ('med', 0.48853754940711464),
   ('prostata', 0.46727549467275492),
   ('schm', 0.45588235294117646),
   ('vulva', 0.44117647058823528),
   ('ovar', 0.40999999999999998),
   ('infekt', 0.40540540540540543),
   ('basis', 0.40059347181008903),
   ('galle', 0.39805825242718446),
   ('nm', 0.38461538461538464),
   ('melanom', 0.37037037037037035),
   ('bew', 0.37037037037037035),
   ('5', 0.36954503249767873),
   ('3', 0.34715025906735753),
   ('mikro', 0.33035714285714285),
   ('hoden', 0.32786885245901637),
   ('op', 0.30959302325581395),
   ('obstrukt', 0.30909090909090908),
   ('alk', 0.30769230769230771),
   ('karz', 0.30681818181818182),
   ('fat', 0.30434782608695654),
   ('progn', 0.28460038986354774),
   ('rauch', 0.28333333333333333),
   ('pet', 0.27777777777777779),
   ('kn', 0.25925925925925924),
   ('endo', 0.25),
   ('ue', 0.24390243902439024),
   ('net', 0.23076923076923078),
   ('marker', 0.22727272727272727),
   ('screen', 0.224),
   ('tu', 0.21621621621621623),
   ('sclc', 0.21052631578947367),
   ('mamma', 0.19881889763779528),
   ('diab', 0.1875),
   ('met', 0.18681318681318682),
   ('6', 0.17962466487935658),
   ('rch', 0.1761006289308176),
   ('gew', 0.16981132075471697),
   ('lymphome', 0.16666666666666666),
   ('krk', 0.16450216450216451),
   ('mrt', 0.15873015873015872),
   ('adj', 0.14981273408239701),
   ('gliom', 0.14634146341463414),
   ('lap', 0.14634146341463414),
   ('str', 0.14344262295081966),
   ('ern', 0.14141414141414141),
   ('alter', 0.13461538461538461),
   ('spaet', 0.13333333333333333),
   ('biops', 0.13186813186813187),
   ('insitu', 0.1276595744680851),
   ('leukaemien', 0.125),
   ('und', 0.12244897959183673),
   ('and', 0.12236286919831224),
   ('8', 0.12195121951219512),
   ('lq', 0.11206896551724138),
   ('po', 0.1111111111111111),
   ('mund', 0.10714285714285714),
   ('gen', 0.096153846153846159),
   ('erk', 0.095588235294117641),
   ('kam', 0.085365853658536592),
   ('neo', 0.071428571428571425),
   ('7', 0.066666666666666666),
   ('2', 0.065727699530516437),
   ('kur', 0.064814814814814811),
   ('sex', 0.057142857142857141),
   ('sono', 0.054054054054054057),
   ('ct', 0.052631578947368418),
   ('makro', 0.04878048780487805),
   ('ln', 0.04878048780487805),
   ('spez', 0.047872340425531915),
   ('haut', 0.045454545454545456),
   ('horm', 0.045454545454545456),
   ('arten', 0.044776119402985072),
   ('frueh', 0.034782608695652174),
   ('bio', 0.027397260273972601),
   ('rez', 0.026315789473684209),
   ('bb', 0.025000000000000001),
   ('nw', 0.020202020202020204),
   ('histo', 0.019230769230769232)]},
```


