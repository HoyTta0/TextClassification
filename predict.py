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


def predictText(text):

    #文本预处理
    comp = re.compile('[^\u4e00-\u9fa5]')
    text = comp.sub('', text)
    lines = open('stopwords-master/百度停用词表.txt', 'r', encoding='utf-8')
    stop_words = [line.strip() for line in lines]
    # print(stop_words)
    word_list = []
    words_list = []
    try:
        words = jieba.cut(text)
        words = [word for word in words if word not in stop_words]
        segmented_words = ','.join(words)
    except AttributeError:
        pass
    finally:
        words_list.append(words)
        word_list.append(segmented_words.strip())
    text = word_list
    print(text)
    #预测
    train_path = "train.pkl"
    with open(train_path, 'rb') as out_data:
        # 按保存变量的顺序加载变量
        X_train = pickle.load(out_data)
    tfidf_vectorizer = TfidfVectorizer()
    train = tfidf_vectorizer.fit_transform(X_train)
    text_test = tfidf_vectorizer.transform(text)
    tfidf_path = 'tfidf.pkl'
    with open(tfidf_path, 'rb') as out_data:
        clf_tfidf = pickle.load(out_data)
    text_predicted = clf_tfidf.predict(text_test)
    print(text_predicted)


# if __name__ == '__main__':
#     text=input()
#     predictText(text)