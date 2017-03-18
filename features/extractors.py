from sklearn.feature_extraction.text import TfidfVectorizer

def abstract_tfidf(df, max_features):
    return TfidfVectorizer(max_features=max_features).fit_transform(features.abstract.fillna(""))


