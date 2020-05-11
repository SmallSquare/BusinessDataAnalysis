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
    This method is for get a list of movie with their information.
    'pages' means how many pages will the spider browse.
    :return: movielist
    """

    moiveList = []

    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_0) AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/71.0.3578.98 Safari/537.36'}
    session = requests.Session()

    # Get movies.
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


def getMovieShortComments(movieid, pages=1):
    """
    This method can get short-comments.
    :return:
    """

    commentList = []

    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_0) AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/71.0.3578.98 Safari/537.36'}
    session = requests.Session()

    # First, try to get the total of comments.
    r = session.get(
        "https://movie.douban.com/subject/" + str(movieid) + "/comments?limit=20&sort=new_score&status=P&start=",
        headers=headers)
    bsObj = bs4.BeautifulSoup(r.text, "html.parser")
    numstr = bsObj.body.find('div', {'id': 'wrapper'}).find('ul', {'class': 'fleft CommentTabs'}) \
        .find('li', {'class': 'is-active'}).span.get_text()
    num = re.match(r'(\D+)(\d+)', numstr)
    total = int(num.group(2))
    print(total)

    # To avoid the situation that the total of comments is less than the number we set.
    if pages * 20 > total:
        pages = int(total / 20 + 1)

    # Get comments.
    for i in range(0, pages):
        r = session.get(
            "https://movie.douban.com/subject/" + str(movieid) + "/comments?limit=20&sort=new_score&status=P&start=" +
            str(i * 20), headers=headers)
        bsObj = bs4.BeautifulSoup(r.text, "html.parser")
        comment_tags = bsObj.body.find('div', {'id': 'comments'}).find_all('div', {'class': 'comment-item'})
        for tag in comment_tags:
            commentList.append(tag.find('p').span.get_text())

    return commentList


getMoviesInfor(20)
getMovieShortComments(30486586, 11)
