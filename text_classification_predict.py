#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
import sklearn
import pandas
from sklearn.feature_extraction.text import TfidfVectorizer , CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from pyvi import ViTokenizer , ViPosTagger
from sklearn import decomposition, ensemble
import pyvi
import gensim
import os 
import re
import unidecode
 
def bo_dau(s):
    return unidecode.unidecode(s)
    

class TextClassificationPredict(object):
    def __init__(self):
        self.test = None

    def get_data(folder_path):
        dir_path = os.path.dirname(os.path.realpath(os.getcwd()))
        dir_path = os.path.join(dir_path, 'Text-Classisfier')   
        train_path = os.path.join(dir_path, 'sentiment_analysis_train.txt')
        data_train_feature = []
        data_train_target = []
        for lines in open(train_path, 'r', encoding="utf8"):
                    words = lines.strip().split()
                    lines = ' '.join(words[1:])
                    lines = gensim.utils.simple_preprocess(lines)
                    lines = ' '.join(lines)
                    lines = ViTokenizer.tokenize(lines)
                    lines = bo_dau(lines)
                    data_train_feature.append(lines)
                    data_train_target.append(words[0])
        return data_train_feature,  data_train_target

    def predict(self):
        def train_model(classifier, X_data, y_data, X_test=None, y_test=None, is_neuralnet=False, n_epochs=3):       
            X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.2, random_state=35)
            
            if is_neuralnet:
                classifier.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=n_epochs, batch_size=512)
                
                val_predictions = classifier.predict(X_val)
                test_predictions = classifier.predict(X_test)
                val_predictions = val_predictions.argmax(axis=-1)
        #         test_predictions = test_predictions.argmax(axis=-1)
            else:
                classifier.fit(X_train, y_train)
            
                train_predictions = classifier.predict(X_train)
                val_predictions = classifier.predict(X_val)
            print("Validation accuracy: ", metrics.accuracy_score(val_predictions, y_val))

        #  train data
        df_train = self.get_data()
        X_data = df_train[0]
        y_data = df_train[1]
        tfidf_vect = TfidfVectorizer(analyzer='word', max_features=30000)
        tfidf_vect.fit(X_data) # learn vocabulary and idf from training set
        X_data_tfidf =  tfidf_vect.transform(X_data)
        model = ensemble.RandomForestClassifier()
        train_model(model, X_data_tfidf, y_data, is_neuralnet=False)
        print('Train xog')
        dir_path = os.path.dirname(os.path.realpath(os.getcwd()))
        dir_path = os.path.join(dir_path, 'Text-Classisfier')   
        old_data = []
        test_data_feature = []
        test_path = os.path.join(dir_path, 'sentiment_analysis_test.txt')
        for lines in open(test_path, 'r', encoding="utf8"):
            text = gensim.utils.simple_preprocess(lines)
            text = ' '.join(text)
            text = ViTokenizer.tokenize(text)
            text = bo_dau(text)
            test_data_feature.append(text)
            old_data.append(lines)
        test_tfidf = tfidf_vect.transform(test_data_feature)
        result = model.predict(test_tfidf)
        result_path =  os.path.join(dir_path, 'result.txt')
        with open(result_path , 'w', encoding='utf-8') as f:
            for id,text in enumerate(result):
                f.writelines(text+ " " + old_data[id])
            f.close()


if __name__ == '__main__':
    tcp = TextClassificationPredict()
    tcp.predict()
    
    