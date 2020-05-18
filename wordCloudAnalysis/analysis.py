# coding=utf-8

# Code by SmallSquare, 2020/5.
# An example that show how to prepare data before wordCloudAnalysis.
# 展示如何在数据分析之前获得数据。

import spider.spider as spider
import database

# Use spider like following.
# 这样用爬虫爬数据。
database.del_all("movie")
database.del_all("comment")
database.insert_movie(spider.getMoviesInfor(10))
database.insert_comment(spider.getMovieShortComments(26885074, 2), 26885074)

# Get data from database.
# 这样获取数据库里的数据。
print(database.get_movies())
print(database.get_comments(26885074))
