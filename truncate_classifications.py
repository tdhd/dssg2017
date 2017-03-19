import numpy as np
import pandas as pd

'''
take top n classifications, to cover ~80% of most common labels
'''

def truncate_classifications(row):
    return map(lambda r: r.replace("'", ""), row.split("',"))

def keep_only_top(orig_list, top_classifications):
    return np.intersect1d(orig_list, top_classifications)

def clean_classifications_and_truncate(df, top_classifications):
    from cleaning_classification_labels import clean_labels
    df['mapped_classification'] = clean_labels.clean_classification(df.classifications.fillna(""), '../data/master/information/translations-labels.csv', 'cleaning_classification_labels/classification_dictionary.csv')
    df['mapped_classification_list'] = df.classifications.str.replace('[\[\]]',"").apply(truncate_classifications)
    df['mapped_classification_list_truncated'] = df['mapped_classification_list'].apply(lambda r: keep_only_top(r, top_classifications))
    return df

def top_labels_from(df):
    from cleaning_classification_labels import clean_labels
    import itertools
    all_classifications = list(itertools.chain(*clean_labels.clean_classification(df.classifications.fillna(""), '../data/master/information/translations-labels.csv', 'cleaning_classification_labels/classification_dictionary.csv')))
    all_classifications = pd.DataFrame({'classification': all_classifications})
    grouped = pd.DataFrame(all_classifications.groupby('classification')['classification'].agg('count'))
    grouped.columns = ['count']
    grouped = grouped.sort_values('count', ascending=False)
    # capture ~82% of most common labels
    return list(grouped[0:400].index)


cleaned = clean_classifications_and_truncate(df, top_labels_from(df))

