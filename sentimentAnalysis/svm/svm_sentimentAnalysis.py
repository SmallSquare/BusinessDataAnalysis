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

# pip install genism -i https://pypi.tuna.tsinghua.edu.cn/simple
import  numpy as np
import  re
import pickle
from sentimentAnalysis.utils import svm_tools as us
from  gensim.models import word2vec
# 短评 搜集的数据有：评论，（评分，有用个数）
# 影评 搜集的数据有：影评，回应， 赞的数目， 踩的数目

class svm_sentimentAnalysis(object):
    def __init__(self, model_path):
        """
        :param model: model path
        """
        self.model_path = model_path
        self.clf = None

        self.initialize()

    def initialize(self):
        with open(self.model_path, 'rb') as file:
            model = pickle.load(file)
            self.clf = model['clf']

    def replace_text(self, text):
        text = re.sub('((https?|ftp|file)://)?[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|].(com|cn)', '', text)
        text = text.replace('\u3000', '').replace('\xa0', '').replace('”', '').replace('"', '')
        text = text.replace(' ', '').replace('↵', '').replace('\n', '').replace('\r', '').replace('\t', '').replace('）',
                                                                                                                    '')
        text_corpus = re.split('[！。？；……;]', text)[0]
        return text_corpus

    def predict_score(self, text_corpus):
        # 分词
        # docs = [un.chinese_word_cut(sentence) for sentence in text_corpus]
        docs = us.svm_chinese_word_cut(text_corpus)
        # model = us.model(docs)
        model = word2vec.Word2Vec.load('../model/svm_train.model')
        vect = us.get_vect(docs, model)
        # 四舍五入，保留三位
        predicted = self.clf.predict_proba(vect)
        print(self.clf.predict(vect))
        print(predicted)

        print('done')
        result = np.around(np.mean(predicted, axis=0), decimals=3)
        return result

    def analyze(self, text):
        text_corpus = self.replace_text(text)
        result = self.predict_score(text_corpus)
        print(result)
        #
        neg = result[0]
        pos = result[1]

        print('差评： {} 好评： {}'.format(neg, pos))



def main():
    model_path = '../model/svm.pkl'

    analyzer = svm_sentimentAnalysis(model_path=model_path)
    # text = '倍感失望的一部诺兰的电影，感觉更像是盗梦帮的一场大杂烩。虽然看之前就知道肯定是一部无法超越前传2的蝙蝠狭，但真心没想到能差到这个地步。节奏的把控的失误和角色的定位模糊绝对是整部影片的硬伤。'
    text = '太难了, 这些算法是什么意思, 情感分析太难了'
    # text = '伍导对自动售卖机有执念，掉地下的那些养乐多捡回家继续喝了吗'
    # text = '太久未有新作的导演往往怀抱过于完整和具象的执念，追求影像的落实，却不再相信变数，此片亦然。本就是童话式漏洞百出的故事设定，而导演恰恰选择了最笨拙的方式——让人物充分表达，恰是如此的表达模式让语言失去了它最迷人的欺骗性。很像是家中长辈极力探听少女的心事，虽然把握到她们笼统的成长方向，却终究无法识别并理解那些真正摄人心魄的悸动瞬间。'
    text = '不好也不坏'
    # text = '太差了'
    # text = '啥玩意儿……………………中学生好词好句300篇'
    # text = '最喜欢的是泡温泉那段，穿得太多是俄罗斯套娃哈哈哈哈哈！Paul妈妈误以为Paul是gay之后的反应好暖心。肚子里没点墨水还追不到心仪的女孩了…不说了滚去读书。伍导演别码代码了，众筹给你拍片（不是）爸爸长得好像尊龙'
    # text = '拍的太好了，太喜欢了'
    analyzer.analyze(text=text)


if __name__ == '__main__':
    main()
    # rand = np.random.randint(0, 2, 1)
    # print(rand[0])