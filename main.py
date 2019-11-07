#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2019/11/5 1:50 PM 

@author: HOY
"""


from model import tfidf
from data_proc import load_data
if __name__ == '__main__':
    # 加载数据
    X_train, X_test, y_train, y_test = load_data()
    # 模型测试
    tfidf_clf=tfidf(X_train,X_test,y_train,y_test)
