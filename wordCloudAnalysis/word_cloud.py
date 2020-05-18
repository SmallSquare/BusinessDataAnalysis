# coding=utf-8

# Code by syy, 2020/5.
# Modified by SmallSquare, 2020/5.
# For making a word cloud.

import os
import jieba
import numpy as np
from PIL import Image
import database
import matplotlib.pyplot as plt
import wordcloud


def show_cloud():
    # Read all of comments.
    total_comment_text = ""
    for comment in database.get_comments(26885074):
        total_comment_text += comment['text']

    # Cut sentences to short words.
    wordlist = jieba.lcut(total_comment_text)
    wordliststr = " ".join(wordlist)

    font = os.path.join(os.path.dirname(__file__), "word_cloud_yahei.ttf")
    mask = np.array(Image.open('word_cloud_mask.png'))  # background and shape of the word cloud
    # Get the word cloud.
    wd = wordcloud.WordCloud(scale=8,
                             width=1920,
                             height=1080,
                             font_path=font,
                             mask=mask,
                             max_font_size=100,
                             min_font_size=12,
                             background_color="white",
                             stopwords=get_stop_words()).generate(wordliststr)
    image_colors = wordcloud.ImageColorGenerator(mask)
    wd.recolor(color_func=image_colors)  # color is from the background image
    plt.figure()
    plt.imshow(wd)
    plt.axis("off")
    plt.show()


def get_stop_words():
    txt_path = os.path.join(os.path.dirname(__file__), 'stopwords.txt')
    f = open(txt_path)
    data_lists = f.readlines()
    stopwords = set()
    for data in data_lists:
        stopwords.add(data.rstrip('\n'))
    return stopwords


if __name__ == '__main__':
    get_stop_words()
    show_cloud()
