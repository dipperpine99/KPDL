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

    def train_model(classifier, X_data, y_data, X_test, y_test, is_neuralnet=False, n_epochs=3):    
        X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.2, random_state=12)
   
        if is_neuralnet:
            classifier.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=n_epochs, batch_size=512)
            
            val_predictions = classifier.predict(X_val)
            test_predictions = classifier.predict(X_test)
            val_predictions = val_predictions.argmax(axis=-1)
            test_predictions = test_predictions.argmax(axis=-1)
        else:
            classifier.fit(X_train, y_train)
        
            train_predictions = classifier.predict(X_train)
            val_predictions = classifier.predict(X_val)
            test_predictions = classifier.predict(X_test)
            
        print("Validation accuracy: ", metrics.accuracy_score(val_predictions, y_val))
        print("Test accuracy: ", metrics.accuracy_score(test_predictions, y_test))

    def get_data(folder_path):
        dir_path = os.path.dirname(os.path.realpath(os.getcwd()))
        dir_path = os.path.join(dir_path, 'Text-Classisfier')   
        train_path = os.path.join(dir_path, 'sentiment_analysis_short_train.txt')
        data_train_feature = []
        data_train_target = []
        for lines in open(train_path, 'r', encoding="utf8"):
                    words = lines.strip().split()
                    text = "".join(words[1:])
                    text = gensim.utils.simple_preprocess(lines)
                    text = ''.join(text)
                    text = ViTokenizer.tokenize(text)
                    text = bo_dau(text)
                    data_train_feature.append(text)
                    data_train_target.append(words[0])
        return data_train_feature,  data_train_target

    def predict(self):

        dir_path = os.path.dirname(os.path.realpath(os.getcwd()))
        dir_path = os.path.join(dir_path, 'Text-Classisfier')   
        #  test data
        old_data = []
        test_data_feature = []
        test_data_target = []
        test_path = os.path.join(dir_path, 'sentiment_analysis_short_test.txt')
        for lines in open(test_path, 'r', encoding="utf8"):
            words = lines.strip().split()
            text = "".join(words[1:])
            text = gensim.utils.simple_preprocess(lines)
            text = ''.join(text)
            text = ViTokenizer.tokenize(text)
            text = bo_dau(text)
            test_data_feature.append(text)
            test_data_target.append(words[0])
            old_data.append(lines)
    
        #  train data
        df_train = self.get_data()
        count_vect = TruncatedSVD(n_components=300, random_state=42)
        count_vect.fit(df_train[0])
        data_train_feature_count = count_vect.transform(df_train[0])
        data_test_feature_count = count_vect.transform(test_data_feature)
        encoder = LabelEncoder()

        data_train_target_encode = encoder.fit_transform(df_train[1])
        data_test_target_encode = encoder.fit_transform(test_data_target)

        self.train_model(linear_model.LogisticRegression(), data_train_feature_count , data_train_target_encode , data_test_feature_count , data_test_target_encode)

        # #  test data
        # test_data_feature = []
        # test_data_target = []

        # # init model naive bayes
        # model = NaiveBayesModel()
 
            # text = gensim.utils.simple_preprocess(lines)
            # text = ' '.join(text)
            # text = ViTokenizer.tokenize(text)
            # text = bo_dau(text)
            # test_data.feature(text)
            # test_data_target.append(
        # df_test = pd.DataFrame(test_data)
        # clf = model.clf.fit(df_train["feature"], df_train.target)
        # predicted = clf.predict(df_test["feature"])
        # result_path = os.path.join(dir_path, 'result1.txt')
        # with open(result_path , 'w', encoding='utf-8') as f:
        #     for id, text in enumerate(old_data):
        #         add_label_text = predicted[id] + " " +text
        #         f.writelines(add_label_text)
        # f.close()
        # Print predicted result
        # print(predicted)
        # print(clf.predict_proba(df_test["feature"]))


if __name__ == '__main__':
    tcp = TextClassificationPredict()
    tcp.predict()
    
    