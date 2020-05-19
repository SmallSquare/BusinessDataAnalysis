# coding=utf-8

# Change by SmallSquare.
# Source code is from the Internet, I changed many details.
# IP地址取自国内免费髙匿代理IP网站：http://www.xicidaili.com/nn/

from bs4 import BeautifulSoup
import requests
import random
import sys


def get_ip_list(url='http://www.xicidaili.com/nn/', headers={
    'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2785.143 Safari/537.36'}):
    ip_list = []
    for page in range(1, 2):
        web_data = requests.get(url + str(page), headers=headers)
        soup = BeautifulSoup(web_data.text, 'lxml')
        ips = soup.find_all('tr')
        for i in range(1, len(ips)):
            ip_info = ips[i]
            tds = ip_info.find_all('td')
            ip_list.append(tds[1].text + ':' + tds[2].text)
    return ip_list


def get_random_ip():
    proxy_list = []
    for ip in get_ip_list():
        proxy_list.append('http://' + ip)
    proxy_ip = random.choice(proxy_list)
    proxies = {'http': proxy_ip}
    return proxies


def get_workable_ip():
    for ip in get_ip_list():
        print("Trying to get a workable proxy ip.")
        try:
            proxies = {'http': "http://" + ip}
            res = requests.get('https://movie.douban.com/', proxies=proxies, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2785.143 Safari/537.36'})
            if "异常请求" in res.text:
                raise BufferError
            print("Found a workable ip: " + str(proxies['http']))
            return proxies
        except Exception as e:
            print(ip + " is not workable. Try to test next proxy ip.")
            print(e)
    print("Fail to get a workable proxy ip, please try again.")
    return None


if __name__ == '__main__':
    print(get_workable_ip())
