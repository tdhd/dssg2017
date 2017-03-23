


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

