#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2019/11/5 11:59 AM 

@author: HOY
"""

import os
import pickle
import pandas as pd
import numpy as np
import re
import jieba
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

def load_data():
    data_path = "datasets.pkl"
    if os.path.exists(data_path):
        print("正在加载已处理后的数据...")
        with open(data_path, 'rb') as out_data:
            # 按保存变量的顺序加载变量
            datasets = pickle.load(out_data)
        return datasets
    else:
        print("正在处理数据。。。")
        # 预处理
        data = pd.read_csv('datasets1.csv', encoding="utf-8", header=None, nrows=6001)
        data.columns = ['n_id', 'id', 'content', 'label']
        data = data.drop(index=[0])

        # 去除空值
        # data['text'].fillna(np.NaN).head(1)
        # data.dropna(inplace=True)
        # print(data.info())

        # 如果正则表达式清洗非中文字符
        # re_data = []
        # for i in data["text"]:
        #     i_re = ''.join(re.findall(r'[\u4e00-\u9fa5]', i))
        #     re_data.append(i_re.strip())
        # data["text"] = re_data

        # 类别标签数值化
        # label = preprocessing.LabelEncoder().fit_transform(data['class_label'])
        # data['class_label'] = label



        # 分词
        # 加载自定义词典
        # jieba.load_userdict('data/dict_out.csv')
        # 加载停用词表
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
        # 划分数据集和测试集
        X_train, X_test, y_train, y_test = \
            train_test_split(data['tokens'], data['label'],stratify=data['label'], test_size=0.2, random_state=1)
        datasets = [X_train, X_test, y_train, y_test]
        # 持久化数据集
        with open(data_path, 'wb') as in_data:
            print("正在保存预处理数据。。。")
            pickle.dump(datasets, in_data, pickle.HIGHEST_PROTOCOL)
            pickle.dump(data, in_data, pickle.HIGHEST_PROTOCOL)
            pickle.dump(words_list, in_data, pickle.HIGHEST_PROTOCOL)
        return datasets