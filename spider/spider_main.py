# coding=utf-8

# Code by SmallSquare, 2020/5.
# Only for the course design of the Business Data Analysis.

import requests
import json
import bs4
import re
import spider.get_proxy as get_proxy


def getMoviesInfor(pages=1, proxy=1):
    """
    This method is for get a list of movie with their information.
    'pages' means how many pages will the spider browse.
    :return: movielist
    """

    moiveList = []

    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_0) AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/71.0.3578.98 Safari/537.36'}
    proxies = None
    if proxy == 1:
        proxies = get_proxy.get_workable_ip()
        if proxies is None:
            print("Didn't find a workable ip, spider will use your true ip to get data.")

    session = requests.Session()

    # Get movies.
    try:
        for i in range(0, pages):
            r = session.get(
                "https://movie.douban.com/j/search_subjects?type=movie&tag=%E7%83%AD%E9%97%A8&sort=recommend&"
                "page_limit=20&page_start=" + str(i * 20),
                headers=headers, proxies=proxies)
            jsondatum = json.loads(r.text)
            for movie in jsondatum['subjects']:
                moiveList.append(
                    {'title': movie['title'], 'rate': movie['rate'], 'id': int(movie['id'])})
            # print(moiveList)
            # print(len(moiveList))
    except AttributeError as e:
        print("Limited by website, please change your proxy.爬虫好像受到网站的限制，请更换代理。")

    return moiveList


def getMovieShortComments(movieid, pages=1, proxy=1):
    """
    This method can get short-comments.
    :return: commentlist
    """

    commentList = []

    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_0) AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/71.0.3578.98 Safari/537.36',
        'Cookie': 'bid=PFXqD9SdoDo; douban-fav-remind=1; gr_user_id=0f03311e-0e28-4e2f-a8fd-3a272d2a525f; _vwo_uuid_v2=D54BE21A153A50F178B1EEA3EE252805F|d0f6410ffbf6226399de9cd1715afb86; viewed="1148282_30329536_25815142"; ll="118172"; push_doumail_num=0; douban-profile-remind=1; __yadk_uid=7QS0r1GHatoz4fkcP2sh8IWeD8YWzQ4u; push_noty_num=0; __utmv=30149280.18600; _ga=GA1.2.449624121.1587021337; __utmc=30149280; __utmz=30149280.1589694675.4.3.utmcsr=m.douban.com|utmccn=(referral)|utmcmd=referral|utmcct=/movie/; __utmc=223695111; __utmz=223695111.1589694675.4.3.utmcsr=m.douban.com|utmccn=(referral)|utmcmd=referral|utmcct=/movie/; __gads=ID=352a53130bca4285:T=1589699239:S=ALNI_MYKpXBWoi1resUvUVMC-9bRu-CuSw; _pk_ref.100001.4cf6=%5B%22%22%2C%22%22%2C1589784625%2C%22https%3A%2F%2Fm.douban.com%2Fmovie%2F%22%5D; _pk_ses.100001.4cf6=*; ap_v=0,6.0; __utma=30149280.449624121.1587021337.1589694675.1589784731.5; __utma=223695111.299663224.1587002697.1589694675.1589784731.5; __utmb=223695111.0.10.1589784731; __utmt=1; __utmb=30149280.1.10.1589784731; dbcl2="186000836:vB8x8LL+q3k"; ck=kTW_; _pk_id.100001.4cf6=ffb676b0890cad74.1587002697.6.1589786159.1589699369.'
    }
    session = requests.Session()

    proxies = None
    if proxy == 1:
        proxies = get_proxy.get_workable_ip()

    # First, try to get the total of comments.
    r = session.get(
        "https://movie.douban.com/subject/" + str(movieid) + "/comments?limit=20&sort=new_score&status=P&start=",
        headers=headers, proxies=proxies)
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
    try:
        for i in range(0, pages):
            r = session.get(
                "https://movie.douban.com/subject/" + str(
                    movieid) + "/comments?limit=20&sort=new_score&status=P&start=" +
                str(i * 20), headers=headers)
            bsObj = bs4.BeautifulSoup(r.text, "html.parser")
            comment_tags = bsObj.body.find('div', {'id': 'comments'}).find_all('div', {'class': 'comment-item'})
            pattern = re.compile('\d{2}')
            for tag in comment_tags:
                temp = {}
                t = tag.find('span', {'class': re.compile('(.*) rating')})
                if t is not None:
                    star = int(pattern.findall(t['class'][0])[0])
                    # print(star)
                    temp['comment'] = tag.find('p').span.get_text()
                    temp['star'] = star
                    commentList.append(temp)
    except AttributeError as e:
        print("Limited by website, please change your proxy.爬虫好像受到网站的限制，请更换代理。")
    return commentList


if __name__ == '__main__':
    # getMoviesInfor(20)
    getMovieShortComments(30486586, 6)
