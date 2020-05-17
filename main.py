# coding=utf-8

# Code by SmallSquare, 2020/5.
#

import spider.spider as spider
import database


def test():
    # database.del_all("movie")
    # database.del_all("comment")
    # database.insert_movie(spider.getMoviesInfor(10))
    # database.insert_comment(spider.getMovieShortComments(32659890, 20), 32659890)
    # # No more than 10 pages. getMovieShortComments()方法不要爬10页以上，因为豆瓣限制了短评页数。
    # print(database.get_movies())
    # print(database.get_comments(26885074))
    pass


if __name__ == '__main__':
    test()
