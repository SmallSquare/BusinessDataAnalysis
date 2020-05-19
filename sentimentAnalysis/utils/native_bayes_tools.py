# coding=utf-8

# Change by CZ.
"""
预处理的一些工具

"""
import  jieba
import  numpy as np

def get_stop_words(stopwords):
    with open(stopwords, encoding='utf-8') as f:
        stopwords = f.read()
    stopwords_list = stopwords.split('\n')
    stopwords = [i for i in stopwords_list]
    return stopwords



def chinese_word_cut(mytext):
    return " ".join(jieba.cut(mytext))

