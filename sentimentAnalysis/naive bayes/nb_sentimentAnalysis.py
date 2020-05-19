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
from sentimentAnalysis.utils import native_bayes_tools as un


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
        s = self.clf.predict(new_tfidf)
        print(s)
        print(predicted)
        # 四舍五入，保留三位
        result = np.around(np.mean(predicted, axis=0), decimals=3)
        return result

    def analyze(self, text):
        text_corpus = self.replace_text(text)
        result = self.predict_score(text_corpus)

        neg = result[0]
        mid = result[1]
        pos = result[2]
        # neg = result[0]
        # pos = result[1]

        print('差： {} 一般：{} 好评： {}'.format(neg, mid, pos))
        # print('差： {} 好评： {}'.format(neg, pos))
        # h = neg*pow(2, 0.5) + pos*pow(20, 0.5) + mid*3# 根号2， 根号20
        # print(h)
        # print('你可能的评分{}'.format(round(h, 1)))


def main():
    model_path = '../model/NB.pkl'
    stopwords_path = '../stopwords/hit_stopwords.txt'

    analyzer = sentimentAnalysis(model_path=model_path, stopwords_path=stopwords_path)
    # text = '倍感失望的一部诺兰的电影，感觉更像是盗梦帮的一场大杂烩。虽然看之前就知道肯定是一部无法超越前传2的蝙蝠狭，但真心没想到能差到这个地步。节奏的把控的失误和角色的定位模糊绝对是整部影片的硬伤。'
    # text = '感谢女主最后冲上去亲她！！！我圆满了'
    # text = '希望会有一个即使知道追不上火车，也会为了多看你一会而奋力奔跑的moron，也希望有那么一个人，为了他/她，你甘愿变成moron。'
    text = '网飞青春题材影视剧的同质化过于严重，本片即是一个非常典型的案例，其框架完全可看作是网飞新近爆款《致所有我曾爱过的男孩》的LGBT变奏。伍思薇为外界称道的《面子》的精华在于，它秉承了与李安“父亲三部曲”相似的品质，对中西文化碰撞有着敏锐的洞悉力。而在这部围绕文化认同基本一致的美国青少年（主角虽为华裔，认知阶段却全程在美国发生）的爱情片里，属于亚裔的叙事反倒被淹没。伍思薇舍弃其移民经验视角的几乎全部优势，拍了一部本质俗不可耐的美式青春片。★★'
    # text = '伍导对自动售卖机有执念，掉地下的那些养乐多捡回家继续喝了吗'
    # text = '太久未有新作的导演往往怀抱过于完整和具象的执念，追求影像的落实，却不再相信变数，此片亦然。本就是童话式漏洞百出的故事设定，而导演恰恰选择了最笨拙的方式——让人物充分表达，恰是如此的表达模式让语言失去了它最迷人的欺骗性。很像是家中长辈极力探听少女的心事，虽然把握到她们笼统的成长方向，却终究无法识别并理解那些真正摄人心魄的悸动瞬间。'
    # text = '不好也不坏'
    # text = '啥玩意儿……………………中学生好词好句300篇'
    analyzer.analyze(text=text)


if __name__ == '__main__':
    main()
