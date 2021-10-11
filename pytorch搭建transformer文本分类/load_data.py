import pandas as pd
import numpy as np
from tqdm import tqdm
#--------------------------加载数据----------------------------
def load_data(config):
    df = pd.read_csv('/新闻文本分类/train_set.csv',sep='\t')

    train = []
    targets = []
    label = df['label'].values
    text = df['text'].values
    id = 0
    vocabs_size = 0
    for val in tqdm(text):
        s = val.split(' ')
        single_data = []
        for i in range(len(s)):
            vocabs_size = max(vocabs_size,int(s[i])+1)
            single_data.append(int(s[i])+1)
            if len(single_data)>=config.pad_size:
                train.append(single_data)
                targets.append(int(label[id]))
                single_data = []
        if len(single_data)>=150:
            single_data = single_data + [0]*(config.pad_size-len(single_data))
            train.append(single_data)
            targets.append(int(label[id]))  
        id += 1
        


    train = np.array(train)
    targets = np.array(targets)
    return train,targets,vocabs_size
