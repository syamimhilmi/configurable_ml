import pandas as pd
import preprocessing as pp
import pickle
import json

import nltk

from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

from flask import request, jsonify

def ratio(st):

    if st == '60:40':
        return 0.4
    elif st == '70:30':
        return 0.3
    elif st == '80:20':
        return 0.2

def train(name,data,path):
    #main router

    #create label map for reference
    pp.createLabelMap(data,path)
    
    if name == 'NB':
        train_NB(data,path)

def classify(data,path):
    #main router
    with open(path + '/config.json', 'r') as file:
            config = json.load(file)

    if config['config']['model_name'] == 'NB':
        return classify_NB(data,path)

def train_NB(data,path):

    from sklearn.naive_bayes import MultinomialNB
    #Import scikit-learn metrics module for accuracy calculation
    from sklearn import metrics
    # Model Generation Using Multinomial Naive Bayes

    with open(path + '/config.json', 'r') as file:
            config = json.load(file)

    test_size = ratio(config['config']['ratio'])

    name = path
    path = path + '/'

    #preprocessing
    data['body'] = [pp.rmpunc(st) for st in data['body']]
    data['title'] = [pp.rmpunc(st) for st in data['title']]
    data['text'] = data['title'] + ' ' + data['body']
    data['normalize'] = pp.malaya_normalize(data['text'])
    data['cleaned'] = pp.malaya_stemming(data['normalize'])

    #to debug
    data.to_csv(path + '/data.csv',index=False)

    #training
    X = data['cleaned']
    y = data['label']

    
    flag = False

    token = RegexpTokenizer(r'[a-zA-Z0-9]+')
    cv_vectorizer = CountVectorizer(input='content', lowercase=True, tokenizer = token.tokenize)
    tfidf_vectorizer = TfidfVectorizer(lowercase=True, tokenizer = token.tokenize, sublinear_tf = True)

    if flag:
        res = tfidf_vectorizer.fit_transform(X)
    else:
        res = cv_vectorizer.fit_transform(X)

    with open(path + 'vectorizer.pkl','wb') as file:
        pickle.dump(res,file)

    X_train, X_test, y_train, y_test = train_test_split(res, y, test_size = test_size, random_state = 1)
    

    selector = SelectKBest(chi2, k='all')
    selector.fit(X_train, y_train)

    clf = MultinomialNB(alpha=0.1).fit(selector.transform(X_train), y_train)
    predicted = clf.predict(X_test)

    #metrics
    num_class = data.label.nunique()
    score_acc = metrics.accuracy_score(y_test, predicted)
    score_f1 = metrics.f1_score(y_test, predicted, average='macro')
    score_precision = metrics.precision_score(y_test,predicted, average='macro')
    score_recall =metrics.recall_score(y_test,predicted, average='macro')

    #result_list = [[num_class,score_acc,score_f1,score_precision,score_recall,test_size]]    
    #results = pd.DataFrame(result_list,columns=['num','accuracy','f1','precision','recall','test_size'])
    result_dict = {'name':name,'num':num_class,'accuracy':score_acc,'f1':score_f1,'precision':score_precision,'recall':score_recall,'test_size':test_size,'total_data':y.shape[0]}

    with open(path + 'metrics.json', 'w') as file:
        json.dump(result_dict,file)
    
    with open(path + 'model.pkl','wb') as file:
        pickle.dump(clf,file)

    with open(path + 'selector.pkl','wb') as file:
        pickle.dump(selector,file)

def classify_NB(input_data,path):

    from sklearn.naive_bayes import MultinomialNB
    #Import scikit-learn metrics module for accuracy calculation

    with open(path + '/model.pkl', 'rb') as file:  
        clf = pickle.load(file)

    with open(path + '/selector.pkl', 'rb') as file:  
        selector = pickle.load(file)

    with open(path + '/vectorizer.pkl', 'rb') as file:  
        vectorizer = pickle.load(file)

    data = pd.read_csv(path + '/data.csv')
    X = data['cleaned']

    flag = False

    token = RegexpTokenizer(r'[a-zA-Z0-9]+')
    cv_vectorizer = CountVectorizer(input='content', lowercase=True, tokenizer = token.tokenize)
    tfidf_vectorizer = TfidfVectorizer(lowercase=True, tokenizer = token.tokenize, sublinear_tf = True)
    
    complaint = input_data['text'] 
    ls = []
    ls.append(complaint)
    df = pd.DataFrame(ls, columns=['text'])

    if flag:
        res = tfidf_vectorizer.fit_transform(X)
    else:
        res = cv_vectorizer.fit_transform(X)
        
    #2 - normalize
    df['normalized'] = pp.malaya_normalize(df['text'])

    #3 - stemming
    df['stemmed'] = pp.malaya_stemming(df['normalized'])

    #4 - remove stop words, punctuation and normalize capitalization
    df['cleaned'] = pp.malaya_rmstop(df['stemmed'])
    complaint = df['cleaned']
    if flag:
        complaint = tfidf_vectorizer.transform(complaint)
    else:
        complaint = cv_vectorizer.transform(complaint)
    predicted = clf.predict(complaint)
    
    return jsonify({'category':predicted[0]})
