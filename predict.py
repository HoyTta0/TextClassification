#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2019/11/7 10:13 AM 

@author: HOY
"""
import pickle
import jieba
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

def predictText(X_train):

    data = pd.read_csv('before.csv', encoding="utf-8", header=None)
    data.columns = ['n_id', 'id', 'content', 'label']
    data = data.drop(index=[0])

    lines = open('stopwords-master/百度停用词表.txt', 'r', encoding='utf-8')
    stop_words = [line.strip() for line in lines]
    # print(stop_words)
    word_list = []
    words_list = []
    for sent in data['content']:
        try:
            words = jieba.cut(sent)
            words = [word for word in words if word not in stop_words]
            segmented_words = ','.join(words)
        except AttributeError:
            continue
        finally:
            words_list.append(words)
            word_list.append(segmented_words.strip())
    data['tokens'] = word_list

    def tfidf(data):
        tfidf_vectorizer = TfidfVectorizer()
        train = tfidf_vectorizer.fit_transform(data)
        return train, tfidf_vectorizer
    # 文本特征提取
    X_train_tfidf, tfidf_vectorizer = tfidf(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(data['tokens'])

    tfidf_path = 'tfidf.pkl'
    with open(tfidf_path, 'rb') as out_data:
        clf_tfidf = pickle.load(out_data)
    y_predicted_tfidf = clf_tfidf.predict(X_test_tfidf)
    data['label'] = y_predicted_tfidf

    data.to_csv('after.csv', encoding="utf_8_sig",columns=[ 'id', 'content', 'label'])
