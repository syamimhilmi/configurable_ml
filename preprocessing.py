import tensorflow as tf
from tensorflow import keras

import numpy as np
import pandas as pd

#PREPROCESSING
import malaya

def rmstop(words, st):
    words = words.split()
    for w in words:
        st = st.replace(w,'')
    return st

def rmpunc(st):
    
    import string
    table = str.maketrans('','',string.punctuation)
    st = [w.translate(table).lower() for w in st.split()]
    #return st
    sst = ''
    for each in st:
        sst = sst + each + ' '
    return sst

def readLabelMap(path):
    labels = {}
    with open(path + '/label.map','r') as f:
        for line in f:
            label = line.split(":")
            labels.update({label[0].strip():label[1].strip()})
    return labels

def createLabelMap(data,path):
    data['map'] = data['label'].astype(str) + ':' + data['category'].astype(str)

    labels = data['map'].unique()
    
    with open(path + '/label.map','w') as file:
        for each in labels:
            file.write(str(each) + '\n')

def getLabel(labels,st):

    for num, label in labels.items():
        if label == st:
            return(num)

def malaya_normalize(df):

    corrector = malaya.spell.probability()
    normalizer = malaya.normalize.normalizer(corrector)
    temp = (normalizer.normalize(res) for res in df)
    ls = []
    for each in temp:
        ls.append(each['normalize'])

    return ls

def malaya_stemming(df):
    model = malaya.stem.deep_model()
    temp = (model.stem(res) for res in df)
    ls = []
    for each in temp:
        ls.append(each)
    return ls

def malaya_rmstop(df):

    f = open('malay.stoplist')
    stopwords = f.read()
    f.close()
    temp = (rmstop(stopwords,st) for st in df)
    ls = []
    for each in temp:
        ls.append(rmpunc(each))

    return ls

if __name__ == '__main__':
    
    data = pd.read_csv('complaints.csv')
    data.columns = ['text','name','label']
    #1 - correct spelling
    ## Problem
        
    #2 - normalize
    data['normalized'] = malaya_normalize(data['text'])

    #3 - stemming
    data['stemmed'] = malaya_stemming(data['normalized'])

    #4 - remove stop words, punctuation and normalize capitalization
    data['cleaned'] = malaya_rmstop(data['stemmed'])

    #Prepare label map for easy reference
    ls = set(data['label'])
    ls = list(ls)
    with open('label.map','w') as out:
        index = 1
        for e in ls:
            out.write(str(str(index)+':'+e+'\n'))
            index += 1
            
    labels = readLabelMap()
    temp = (getLabel(labels,res) for res in data['label'])
    ls = []
    for each in temp:
        ls.append(each)

    data['class'] = ls

    data.to_csv('preprocessed.csv', index=False)
