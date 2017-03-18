def get_Label_Top_level(data):
    
    def to_list(x):
        res = list()
        
        for ii in x: 
            try: 
                int(ii)
                res.append(ii)
            except: 
                pass
        return res
    
    labelVectorizer = MultiLabelBinarizer()

    m = data["label_top_level"].apply(to_list)
    y = labelVectorizer.fit_transform(m)
    return y
    
#get_Label_Top_level(data)

