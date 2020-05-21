# coding=utf-8

# code by Cz.
import pickle
import numpy as np
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences


if __name__ == '__main__':

    ####################################################################
    # tokenizer = text.Tokenizer(num_words=147, lower=True, split=' ')
    # tokenizer.fit_on_texts(text)
    # print(tokenizer.word_index)
    # x_to_seq = tokenizer.texts_to_sequences(text)
    # print('x to seq shape', x_to_seq)
    # 二维张量
    # X = sequence.pad_sequences(x_to_seq, maxlen=147, padding='post', value=0)
    ####################################################################

    # text = '倍感失望的一部诺兰的电影，感觉更像是盗梦帮的一场大杂烩。虽然看之前就知道肯定是一部无法超越前传2的蝙蝠狭，但真心没想到能差到这个地步。节奏的把控的失误和角色的定位模糊绝对是整部影片的硬伤。'
    # text = '啥玩意儿……………………中学生好词好句300篇'
    # text = '剧情一般，演技一般，场景一般，总体一般'
    # text = '最喜欢的是泡温泉那段，穿得太多是俄罗斯套娃哈哈哈哈哈！Paul妈妈误以为Paul是gay之后的反应好暖心。肚子里没点墨水还追不到心仪的女孩了…不说了滚去读书。伍导演别码代码了，众筹给你拍片（不是）爸爸长得好像尊龙'
    text = '拍的太好了，太喜欢了'
    # text = '电影太差了，一点也看不下去，没有剧情'
    # text =  text = '太久未有新作的导演往往怀抱过于完整和具象的执念，追求影像的落实，却不再相信变数，此片亦然。本就是童话式漏洞百出的故事设定，而导演恰恰选择了最笨拙的方式——让人物充分表达，恰是如此的表达模式让语言失去了它最迷人的欺骗性。很像是家中长辈极力探听少女的心事，虽然把握到她们笼统的成长方向，却终究无法识别并理解那些真正摄人心魄的悸动瞬间。'

    # 导入字典
    with open('../data/word_dict.pk', 'rb') as f:
        word_dictionary = pickle.load(f)
    with open('../data/label_dict.pk', 'rb') as f:
        output_dictionary = pickle.load(f)
    try:
        input_shape = 300
        # text = "电视刚安装好，说实话，画质不怎么样，很差！"
        x = [[word_dictionary[word] for word in text]]
        x = pad_sequences(maxlen=input_shape, sequences=x, padding='post', value=0)
        model = load_model('../model/lstm_3_r.h5')
        y_predict = model.predict(x)
        print(y_predict)
        label_dict = {v: k for k, v in output_dictionary.items()}
        print('输入语句: %s' % text)
        print('情感预测结果: %s' % label_dict[np.argmax(y_predict)])
    except KeyError as err:
        print("您输入的句子有汉字不在词汇表中，请重新输入！")
        print("不在词汇表中的单词为：%s." % err)

