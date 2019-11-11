#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2019/11/5 2:33 PM 

@author: HOY
"""
import pandas as pd
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

comp = re.compile('[^\u4e00-\u9fa5]')
data = pd.read_csv('non_labeled_datasets.csv',usecols=['id','title'], encoding = 'gbk')
# data = data.dropna()
# re_data = []
# for i in data['title']:
#     i_re = comp.sub('', i)
#     re_data.append(i_re.strip())
# data['title'] = re_data
data = data.dropna(how='any')
data.to_csv('datasets.csv',encoding="utf_8_sig")

# data = pd.read_csv('datasets1.csv', encoding='utf-8')
# labels_list = []
# train_path = "train.pkl"
# with open(train_path, 'rb') as out_data:
#     # 按保存变量的顺序加载变量
#     X_train = pickle.load(out_data)
# tfidf_vectorizer = TfidfVectorizer()
# train = tfidf_vectorizer.fit_transform(X_train)
# # text_test = tfidf_vectorizer.transform(text)
# tfidf_path = 'tfidf.pkl'
# with open(tfidf_path, 'rb') as out_data:
#     clf_tfidf = pickle.load(out_data)
#
# for sent in data['title']:
#         text_test = tfidf_vectorizer.transform([sent])
#         label=clf_tfidf.predict(text_test)
#         labels_list.append(label)
# data['label']=labels_list
# data.to_csv('datasets1.csv',encoding="utf_8_sig")
# print(data)

# data_path = "datasets.pkl"
# with open(data_path, 'rb') as out_data:
#     # 按保存变量的顺序加载变量
#     datasets = pickle.load(out_data)
#     data = pickle.load(out_data)
# print(datasets)
