#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2019/11/4 2:52 PM 

@author: HOY
"""
import os
import pandas as pd
import re
dirs = './data/'
files = os.walk(dirs)
dirss = [ dirss for root,dirss,files in os.walk(dirs)][0]
files = [ files for root,dirss,files in os.walk(dirs)]
id = []
content = []
label = []
comp = re.compile('[^A-Z^a-z^0-9^\u4e00-\u9fa5]')
for i in range(0,len(dirss)):
    for zz in files[i+1]:
        if zz.find('.txt')>0:
            with open('data/'+dirss[i]+'/'+zz,'r') as f:
                ff = f.read()
                ff = comp.sub('', ff)
                id.append(zz.split('.txt')[0])
                content.append(ff)
                label.append(int(dirss[i]=="教育"))
data = pd.DataFrame([id,content,label])
# data = data.sample(n=41936,axis=1)
data = data.T
data.to_csv('data.csv',encoding="utf_8_sig")


# fr = open('data_psample.csv','rb').read()
# with open('result.csv','ab') as f: #将结果保存为result.csv
#     f.write(fr)
# fr = open('data_nsample.csv','rb').read()
# with open('result.csv','ab') as f: #将结果保存为result.csv
#     f.write(fr)

# df = pd.read_csv("result.csv",header=0)
# df = df.drop(index=[41936])
# df=df.reset_index(drop=True)
# df.to_csv("data.csv", header = True,index=False,encoding="utf_8_sig")
