from ris import read_ris_lines
from ovr import labels_of, features_of, classify_cancer, clean_kws
from views import df_from
import numpy as np
import pandas as pd

with open('/home/ppschmidt/dssg2017/dssg2017/data/information-fuer-study-tagger.ris') as f:
    df = df_from('\n'.join(f.readlines()))

print(df.shape)

def split_kws_by_comma(row_kws):
    l = map(lambda kw: kw.lower().split(','), row_kws)
    return [item for sublist in l for item in sublist]

# clean keywords
df['KW'] = df.KW.apply(clean_kws)
# split all keywords by comma to produce also single word keywords
# e.g. 'harn' instead of 'harn,1-karz'
df['KW'] = df.KW.apply(split_kws_by_comma)
X = features_of(df)
# take p% of most common labels
y, label_names = labels_of(df, 'KW', p = 0.90)

print(X.shape, y.shape)
print(label_names)

# model selection
clf = classify_cancer(X, y, label_names)

# load test data
with open('/home/ppschmidt/dssg2017/dssg2017/data/ovar-update-201612.ris') as f:
    df = df_from('\n'.join(f.readlines()))
X = features_of(df)

y = clf.predict_proba(X)

print(df.shape, y.shape)
# label probas in %
df = pd.concat([pd.DataFrame(y*100), df.T1, df.N2], axis=1)
df.columns = np.concatenate((label_names, ['title', 'abstract']))
df.to_csv('ovar-update-201612.ris.csv', float_format='%.2f', index=False)
