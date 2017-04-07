import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

breast_cancer = datasets.load_breast_cancer()
data = breast_cancer['data']
target = breast_cancer.target
iterations = 500

l1 = 0
l2 = 0
l3 = 0

# branch micha

for i in range(iterations):
    X, y = shuffle(data, target)
    grd_lm = LogisticRegression()
    sgd = SGDClassifier(alpha=0.01, n_iter=100)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
    # It is important to train the ensemble of trees on a different subset
    # of the training data than the linear regression model to avoid
    # overfitting, in particular if the total number of leaves is
    # similar to the number of training samples
    X_train, X_train_lr, y_train, y_train_lr = train_test_split(X_train,
                                                                y_train,
                                                                test_size=0.5)

    mod_sgd = sgd.fit(X=X_train,y=y_train)

    #logistic regression
    mod_lr = grd_lm.fit(X=X_train,y=y_train)

    #embed features in space with gbt
    grd = GradientBoostingClassifier(n_estimators=10)
    grd.fit(X_train, y_train)
    #Hot encoding of the resulting leaves for each sample
    grd_enc = OneHotEncoder()
    grd_enc.fit(grd.apply(X_train)[:, :, 0])

    #create new dataset consisting of old features and the hot encoded gbt result
    X_train_enr_lr = np.hstack(
            (X_train_lr,
             grd_enc.transform(grd.apply(X_train_lr)[:,:,0]).toarray())
    )



    grd_lm = LogisticRegression()


    grd_lm.fit(X_train_enr_lr, y_train_lr)


    #use trained gbt and enrich dataset
    X_test_enr = np.hstack(
                            (X_test,grd_enc.transform(grd.apply(X_test)[:, :, 0]).toarray())
                        )

    y_pred_grd_lm = grd_lm.predict_proba(
                    X_test_enr
                    )[:, 1]

    gbt_loss = np.sum((y_test - grd_lm.predict(X_test_enr))**2)

    lr_loss =  np.sum((y_test - mod_lr.predict(X_test))**2)
    sgd_loss =  np.sum((y_test - mod_sgd.predict(X_test))**2)

    l1 = l1 + gbt_loss
    l2 = l2 + lr_loss
    l3 = l3 + sgd_loss


    #print sgd_loss
    #print gbt_loss
    #print lr_loss

print 100 - (l1/float(iterations))

print 100 - (l2/float(iterations))

print 100 - (l3/float(iterations))


#print np.round(y_pred_grd_lm,0)
fpr_grd_lm, tpr_grd_lm, _ = roc_curve(y_test, y_pred_grd_lm)
plt.plot(fpr_grd_lm, tpr_grd_lm, label='GBT + LR')
#plt.show(block=True)
