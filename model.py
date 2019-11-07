#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2019/11/5 10:44 AM 

@author: HOY
"""

import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

def get_metrics(y_test,y_predicted):
    """
    :param y_test: 真实值
    :param y_predicted: 预测值
    :return:
    """
    # 精确度=真阳性/（真阳性+假阳性）
    precision=precision_score(y_test,y_predicted,pos_label=None,average='weighted')
    # 召回率=真阳性/（真阳性+假阴性）
    recall=recall_score(y_test,y_predicted,pos_label=None,average='weighted')

    # F1
    f1=f1_score(y_test,y_predicted,pos_label=None,average='weighted')

    # 精确率
    accuracy=accuracy_score(y_test,y_predicted)
    return accuracy,precision,recall,f1

def tfidf(X_train,X_test,y_train,y_test):

    def tfidf(data):
        tfidf_vectorizer = TfidfVectorizer()
        train = tfidf_vectorizer.fit_transform(data)
        return train, tfidf_vectorizer
    # 文本特征提取
    X_train_tfidf, tfidf_vectorizer = tfidf(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    print(X_train)
    tfidf_path = 'tfidf.pkl'
    if os.path.exists(tfidf_path):
        print("正在加载已经训练的模型...")
        with open(tfidf_path, 'rb') as out_data:
            clf_tfidf = pickle.load(out_data)
    else:
        print("正在训练tfidf模型...")
        # 声明模型
        clf_tfidf = LogisticRegression(C=10.0, solver='newton-cg', multi_class='multinomial', n_jobs=-1)
        # 训练
        clf_tfidf.fit(X_train_tfidf, y_train)
        # 保存训练的模型
        with open(tfidf_path, 'wb') as in_data:
            pickle.dump(clf_tfidf, in_data, pickle.HIGHEST_PROTOCOL)
            print("tfidf model saved:" + tfidf_path)

    # 预测结果
    y_predicted_tfidf = clf_tfidf.predict(X_test_tfidf)
    ##模型评估
    accuracy_tfidf, precision_tfidf, recall_tfidf, f1_tfidf = get_metrics(y_test, y_predicted_tfidf)
    print("accuracy = %.6f, precision = %.6f, recall = %.6f, f1 = %.6f" % (
    accuracy_tfidf, precision_tfidf, recall_tfidf, f1_tfidf))

    return clf_tfidf