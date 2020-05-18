# coding=utf-8

# Code by Cz, 2020/5.
# 以电影的主题、意义、叙事、角色、视点、镜头、剪辑、声音、动画、特效
# 叙事，角色，镜头，画面构成， 配乐，最新科技等

# 六种常见的评写电影的方法：
# 电影史：从历史发展的角度出发，探索电影自身之间，或与制作条件、受众的关系；
# 民族电影：从文化和民族性格的角度来讨论电影；
# 类型：根据形式和内容的模式对电影进行分类；
# 作者论：通过将影片和导演、演员相联系来定义和检视一部电影
# 形式主义的种类：研究一部影片的风格、结构和技巧等形式的组织方式；
# 意识形态：包含了政治、种族、阶级、性别、家庭等角度。

# pip install jieba -i https://pypi.tuna.tsinghua.edu.cn/simple
import numpy as np
import re
import pickle
import utils.native_bayes_tools as un


# 短评 搜集的数据有：评论，（评分，有用个数）
# 影评 搜集的数据有：影评，回应， 赞的数目， 踩的数目

class sentimentAnalysis(object):
    def __init__(self, model_path, stopwords_path):
        """
        :param model: model path
        """
        self.model_path = model_path
        self.stopwords_path = stopwords_path
        self.stop_words = []
        self.clf = None
        self.vect = None
        self.tfidftransformer = None

        self.initialize()

    def initialize(self):
        self.stop_words = un.get_stop_words(self.stopwords_path)
        with open(self.model_path, 'rb') as file:
            model = pickle.load(file)
            self.clf = model['clf']
            self.vectorizer = model['vectorizer']
            self.tfidftransformer = model['tfidftransformer']

    def replace_text(self, text):
        text = re.sub('((https?|ftp|file)://)?[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|].(com|cn)', '', text)
        text = text.replace('\u3000', '').replace('\xa0', '').replace('”', '').replace('"', '')
        text = text.replace(' ', '').replace('↵', '').replace('\n', '').replace('\r', '').replace('\t', '').replace('）',
                                                                                                                    '')
        text_corpus = re.split('[！。？；……;]', text)
        return text_corpus

    def predict_score(self, text_corpus):
        # 分词
        docs = [un.chinese_word_cut(sentence) for sentence in text_corpus]
        new_tfidf = self.tfidftransformer.transform(self.vectorizer.transform(docs))
        predicted = self.clf.predict_proba(new_tfidf)
        # 四舍五入，保留三位
        result = np.around(predicted, decimals=3)
        return result

    def analyze(self, text):
        text_corpus = self.replace_text(text)
        result = self.predict_score(text_corpus)

        neg = result[0][0]
        pos = result[0][1]

        print('差评： {} 好评： {}'.format(neg, pos))


def main():
    model_path = './model/NB.pkl'
    stopwords_path = 'hit_stopwords.txt'

    analyzer = sentimentAnalysis(model_path=model_path, stopwords_path=stopwords_path)
    text = '倍感失望的一部诺兰的电影，感觉更像是盗梦帮的一场大杂烩。虽然看之前就知道肯定是一部无法超越前传2的蝙蝠狭，但真心没想到能差到这个地步。节奏的把控的失误和角色的定位模糊绝对是整部影片的硬伤。'
    analyzer.analyze(text=text)


if __name__ == '__main__':
    main()
