# coding=utf-8

# Code by Cz, 2020/5.17

import spider.spider_main as sp
import  pandas as pd
import utils.csv_tools as t_csv

def getAllComments():
     movie_id = 'movie_id.csv'
     df = pd.read_csv(movie_id)
     for i in df['id']:
         print(i)
         commentList = sp.getMovieShortComments(i, 24)# 很难控制这个页面
         t_csv.list_to_csv(commentList, 'comments.csv')

if __name__ == '__main__':
    getAllComments()