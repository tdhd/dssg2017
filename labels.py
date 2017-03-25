
def classifications_list_to_cancer_paper_subs_pruned(row):
    '''
    apply this function to the already parsed classifications list (output of labels.to_list)

    df['classifications_list'] = df.apply(lambda row: to_list(row, 'classifications'), axis=1)
    df['cancer_paper_subs_pruned'] = df['classifications_list'].apply(classifications_list_to_cancer_paper_subs_pruned)

    this function converts each entry in the classification list
    to separate cancer type(s), paper type and subcategories
    where it only takes the first of the subcategories and prunes the remaining
    '''
    import re
    def cancers_from(c):
        s = re.search('(.*),\d.*', c, re.IGNORECASE)
        if s:
            return s.group(1).split(",")
        else:
            return ['unknown']
    def papers_from(c):
        s = re.search('.*,(\d).*', c, re.IGNORECASE)
        if s:
            return s.group(1)
        else:
            return 'unknown'

    def cats_from(c):
        s = re.search('.*,\d-(.*)', c, re.IGNORECASE)
        if s:
            return s.group(1)
        else:
            return 'unknown'

    def prune_cats(c):
        return c.split("-")[0]

    # can be multiple cancers
    cancers = map(cancers_from, row)
    cancers = [item for sublist in cancers for item in sublist]
    papers = map(papers_from, row)
    cats = map(cats_from, row)
    pruned_cats = map(prune_cats, cats)
    # print cancers, papers, cats, pruned_cats
    # print cancers + papers + pruned_cats
    return cancers + papers + pruned_cats

def cancer_types_from(classifications_list):
    '''
    given classifications list, extract (possibly more than one) cancer type
    use like:
        df['classifications_list'] = df.apply(lambda row: to_list(row, 'classifications'), axis=1)
        df['classifications_list'].apply(cancer_types_from)
    '''
    import re
    def extract(t):
        s = re.search('(.*),\d.*', t, re.IGNORECASE)
        if s:
            return s.group(1).split(",")
        else:
            return ['unknown']
    l = map(extract, classifications_list)
    l = [item for sublist in l for item in sublist]
    return list(set(l))

def paper_types_from(row):
    '''
    extract paper types from classifications_list column
    df['classifications_list'] = df.apply(lambda row: to_list(row, 'classifications'), axis=1)
    df['paper_type'] = df['classifications_list'].apply(paper_types_from)
    '''
    import re
    def paper_from(token):
        r = re.search('.*(\d).*', token, re.IGNORECASE)
        if r:
            return r.group(1)
        else:
            return 'unknown'

    return map(paper_from, row)

def classifications_list_to_cancer_paper_subs(row):
    '''
    apply this function to the already parsed classifications list (output of labels.to_list)

    df['classifications_list'] = df.apply(lambda row: to_list(row, 'classifications'), axis=1)
    df['cancer_paper_subs'] = df['classifications_list'].apply(classifications_list_to_cancer_paper_subs)

    this function converts each entry in the classification list
    to separate cancer type(s), paper type and subcategories
    '''
    # split cancer types from papertype-subs
    l = map(lambda e: e.split(","), row)
    l = [item for sublist in l for item in sublist]
    # split paper type from subs
    l = map(lambda e: e.split("-"), l)
    l = [item for sublist in l for item in sublist]
    return l

def to_list(row, column):
    '''
    use like df.apply(lambda row: to_list(row, 'classifications'), axis=1)
    to transform classifications from string to list of string
    '''
    cleaned = map(lambda t: t.replace("'", ""), row[column].replace("[", "").replace("]", "").split("',"))
    return filter(lambda r: len(r) > 0, cleaned)

def flat_map(df, column):
    '''
    take a column from df and flatten a list of any type in the column to a single list
    '''
    import itertools
    return list(itertools.chain(*df[column]))

