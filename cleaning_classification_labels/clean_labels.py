import pandas as pd
import numpy as np

#path_for_translation_labels = 'cancer_data/information/translations-labels.csv'   #path of the .csv translation file provided by the data ambassadors
#path_for_manual_transtable = 'classification_dictionary.csv'        # path of the .csv file   that was filled manually by Marie

def clean_classification(class_series, trans_path, manual_trans_path):

    brac_free = class_series.apply(tokenizeCancerLabels)
    
    trans_labels_df = pd.read_csv(trans_path,delimiter=';')

    labels1st = np.unique(trans_labels_df['Label (1st Level)'])

    class_trans_df = pd.read_csv(manual_trans_path,delimiter=';',header=None)

    #### create dictionary that translates to correct first level 
    include_1st_level_dict = {'1-bew': '1-koerper', 
                              '1-gew': '1-koerper',
                              '1-gen': '1-koerper', 
                              '1-horm':'1-koerper',
                              '1-rauch': '1-ps', 
                              '1-alk': '1-ps', 
                              '1-canna': '1-ps',
                              '1-diab': '1-erk',
                              '1-infekt': '1-erk', 
                              '3-tu-marker': '3-lab',
                              '3-biops': '3-lab',
                               '5-fr\xc3\xbch': '5'}

    for label in labels1st:
        include_1st_level_dict[label] = label

    for i in range(class_trans_df.shape[0]):
        cc = class_trans_df.iloc[i]        
        include_1st_level_dict[cc[0]] = cc[1]

    ## done creating the dictionary
    cleaned_series = brac_free.apply(lambda x: clean_levels(x,include_1st_level_dict))

    return cleaned_series


def tokenizeCancerLabels(s):
    '''
    Tokenize the label string and remove empty strings
    '''
    ## if string is an empty list return an empty list
    if s == '[]':
        return []

    ## else return list with    bodypart,classify
    s = s.replace("['","").replace("']","")
    return [t for t in s.split("','") if len(t)>0]



def clean_levels(s_list,trans_dict):
    '''
    function that cleans the label
    '''
    new_list = []
    ## check if there is an entry
    if len(s_list) == 0:
        return new_list
        
    for t in s_list:        
        ts = t.split(',')
        if len(ts) > 2:  ## here cancer applies to more than one bodyparts.. sort them alphabetically
            bodystring = ','.join(sorted(map(lambda x: x.lower(),ts[:-1])))        
            old_class = ts[-1]                        
        else:   #otherwise there is only one bodypart
            bodypart = ts[0]
            old_class = ts[1]
            
        ### correct the classification of the label of format  X-str-...
        ocs = old_class.split('-')
        
        if len(ocs) == 1:
            new_class = old_class
            
        elif len(ocs) == 2:
            if trans_dict.has_key(old_class):
                new_class = trans_dict[old_class]
            else:
                new_class = old_class

        elif len(ocs) > 2:
            first = ocs[0] + '-' + ocs[1]            
            first_extended = ocs[0] + '-' + ocs[1] + '-' + ocs[2]  ## need extra check as '3-tu-marker ' should be maped to  '3-labl

            if trans_dict.has_key('-'.join(ocs)):
                new_class = trans_dict['-'.join(ocs)]
            elif trans_dict.has_key(first_extended):
                new_class = trans_dict[first_extended]
            elif trans_dict.has_key(first):
                new_class = trans_dict[first]
            else:
                new_class = old_class
                    
        new_list.append(ts[0] + ',' + new_class)

    return new_list
         

