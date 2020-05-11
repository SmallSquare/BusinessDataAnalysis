# coding=utf-8

# Code by SmallSquare, 2020/5.
# Only for the course design of the Business Data Analysis.

import time
import requests
import json
import bs4
import re


def getMoviesInfor(pages=1):
    """
    This function is for get a list of movie with their information.
    'pages' means how many pages will the spider browse.
    """

    moiveList = []

    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_0) AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/71.0.3578.98 Safari/537.36'}
    session = requests.Session()

    for i in range(0, pages):
        r = session.get(
            "https://movie.douban.com/j/search_subjects?type=movie&tag=%E7%83%AD%E9%97%A8&sort=recommend&"
            "page_limit=20&page_start=" + str(i),
            headers=headers)
        jsondatum = json.loads(r.text)
        for movie in jsondatum['subjects']:
            moiveList.append(
                {'title': movie['title'], 'rate': movie['rate'], 'id': movie['id'], 'title': movie['title']})

    print(moiveList)
    print(len(moiveList))
    return moiveList


getMoviesInfor(20)
