# coding=utf-8

# Code by SmallSquare, 2020/5.
#

import spider.spider as spider
import database

database.del_all("movie")
database.del_all("comment")
database.insert_movie(spider.getMoviesInfor(10))
database.insert_comment(spider.getMovieShortComments(26885074, 2), 26885074)
