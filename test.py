#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2019/11/5 2:33 PM 

@author: HOY
"""
import pandas as pd
import jieba
from sklearn.model_selection import train_test_split
import pickle
# data = pd.read_csv('data.csv', encoding="utf-8", header=None,nrows=10)
# data.columns = ['n_id', 'id', 'content', 'label']
# lines = open('stopwords-master/百度停用词表.txt', 'r', encoding='utf-8')
# data=data.drop(index=[0])
# stop_words = [line.strip() for line in lines]
# word_list = []
# words_list = []
# for sent in data['content']:
#     try:
#         words = jieba.cut(sent)
#         words = [word for word in words if word not in stop_words]
#         segmented_words = ','.join(words)
#     except AttributeError:
#         continue
#     finally:
#         words_list.append(words)
#         word_list.append(segmented_words.strip())
# data['tokens'] = word_list
#
# X_train, X_test, y_train, y_test = \
#     train_test_split(data['tokens'], data['label'], test_size=0.2, random_state=1)
# datasets = [X_train, X_test, y_train, y_test]
# print(datasets)

data_path = "datasets.pkl"
with open(data_path, 'rb') as out_data:
    # 按保存变量的顺序加载变量
    datasets = pickle.load(out_data)
    data = pickle.load(out_data)
print(datasets)
