import pandas as pd
import numpy as np
import re
import nltk
import json
import os
import pickle
import shutil

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2 

#custom imports
import preprocessing as pp
import models as md


from flask import Flask, request, jsonify
from flask_restful import Resource, Api

def get_top_n_words(corpus, n=None):
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in     vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

def swap_active(new_path):

    with open('active_model.json', 'r') as file:
        old = json.load(file)

    new_dict = {'active' : new_path}
    with open('active_model.json', 'w') as file:
        json.dump(new_dict,file)

    old_path = old['active']
    shutil.rmtree(old_path)

def get_active():

    development = True

    if development:
        with open('active_model.json', 'r') as file:
            return json.load(file)
    else:
        with open('active_model.json', 'r') as file:
            active = json.load(file)
        
        root = '/home/moontaeil/ac/'
        active['active'] = root + active['active']
        return active
    

app = Flask(__name__)
api = Api(app)

from sklearn.naive_bayes import MultinomialNB
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Generation Using Multinomial Naive Bayes


with open('NB.pkl', 'rb') as file:  
    clf = pickle.load(file)

with open('Selector.pkl', 'rb') as file:  
    selector = pickle.load(file)

with open('Vectorizer.pkl', 'rb') as file:  
    vectorizer = pickle.load(file)

data = pd.read_csv('preprocessed.csv')
X = data['cleaned']

class HelloWorld(Resource):
    def get(self):
        return {'Status': 'active'}

class Classify(Resource):
    def post(self):

        input_json = request.get_json(force=True)

        flag = False

        token = RegexpTokenizer(r'[a-zA-Z0-9]+')
        cv_vectorizer = CountVectorizer(input='content', lowercase=True, stop_words='english', tokenizer = token.tokenize)
        tfidf_vectorizer = TfidfVectorizer(lowercase=True, stop_words='english', tokenizer = token.tokenize, sublinear_tf = True)
        
        complaint = input_json['text'] 
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

        labels = readLabelMap()

        return jsonify({'category':translate(labels[str(predicted[0])])})

class List(Resource):
    def get(self):

        ls = []
        with open('models_list.txt', 'r') as file:  
            for line in file:
                ls.append(line.rstrip())

        return jsonify({'names':[ls]})

class WordCloud(Resource):
    def get(self):

        #ncat = top X categories
        #ntop = top X words
        active = get_active()

        data = pd.read_csv(active['active'] + '/data.csv')

        df = pd.DataFrame(data.label.value_counts())
        df.columns = ['label']
        topcat = df.index[:3]

        mapper = pp.readLabelMap(active['active'])

        output = {}

        for e in topcat:
            
            inner = {}
            temp = data[data['label'] == e]
            res = get_top_n_words(temp['text'],5)

            for i in res:
                inner[i[0]] = int(i[1])
            output[mapper[str(e)]] = inner

        return jsonify(output)


class Train(Resource):
    def post(self):

        from datetime import datetime
        
        input_json = request.get_json(force=True)

        name = input_json['config']['model_name']
        path = name + '_' + str(int(datetime.timestamp(datetime.now())))
        os.makedirs(path , exist_ok=True)
        data = input_json['config']['data']
        data = pd.DataFrame(data)
        #data.to_csv(path + '/data.csv')

        with open(path + '/config.json', 'w') as file:
            json.dump(input_json,file)

        md.train(name,data,path)

        swap_active(path)

class ClassifyV2(Resource):
    def post(self):

        input_json = request.get_json(force=True)

        active = get_active()

        return md.classify(input_json,active['active'])

class Status(Resource):
    def get(self):

        resp = {}

        with open('status.txt','r') as file:
            for line in file:
                resp['status'] = line

        return jsonify(resp)

class Metrics(Resource):
    def get(self):

        active = get_active()

        with open(active['active'] + '/metrics.json') as file:
            metrics = json.load(file)

        return metrics        

api.add_resource(HelloWorld, '/')
api.add_resource(Classify,'/classify/')
api.add_resource(List,'/list/')
api.add_resource(WordCloud,'/wordcloud/')
api.add_resource(Train,'/train/')
api.add_resource(Status,'/status/')
api.add_resource(ClassifyV2,'/classifyv2/')
api.add_resource(Metrics,'/metrics/')

if __name__ == '__main__':
    
    app.run(debug=True)
