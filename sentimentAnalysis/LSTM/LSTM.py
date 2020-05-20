# coding=utf-8

# code by Cz.
import pandas as pd
import numpy as np
import keras
import pickle

from keras.models import Sequential
from keras.models import load_model
from keras.utils import np_utils, plot_model
from keras.preprocessing import text, sequence
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split

from sentimentAnalysis.utils import svm_tools as us, common as uc

import matplotlib.pyplot as plt
from matplotlib import font_manager
from itertools import accumulate

class LSTM_U(object):
    def __init__(self, data):
        self.data = pd.read_csv(data)
        self.model = None
        self.vocab_size = None
        self.X = self.y = None
        self.X_train = self.X_test = None
        self.y_train = self.y_test = None

    def pro_precess(self):
        self.data['cut_comment'] = self.data.comment.apply(uc.chinese_word_cut)
        X = self.data['comment']
        self.data['sentiment'] = ['positive' if (x > 30) else 'negtive' for x in self.data['star']]
        y = self.data['sentiment']

        ##########################################################################
        # tokenizer = text.Tokenizer(num_words=300, lower=True, split=' ')
        # tokenizer.fit_on_texts(X)
        # print(tokenizer.word_index)
        # x_to_seq = tokenizer.texts_to_sequences(X)
        # print('x to seq shape', x_to_seq)
        # 二维张量
        # X = sequence.pad_sequences(x_to_seq, maxlen=300, padding='post', value=0)
        # print('X shape', X.shape)
        # y = pd.get_dummies(y).values
        #########################################################################

        # 标签及词汇表
        labels, vocabulary = list(y.unique()), list(X.unique())

        # 构造字符级别的特征
        string = ''
        for word in vocabulary:
            string += word

        vocabulary = set(string)

        # 字典列表
        word_dictionary = {word: i + 1 for i, word in enumerate(vocabulary)}
        with open('../model/word_dict.pk', 'wb') as f:
            pickle.dump(word_dictionary, f)
        inverse_word_dictionary = {i + 1: word for i, word in enumerate(vocabulary)}
        label_dictionary = {label: i for i, label in enumerate(labels)}
        with open('../model/label_dict.pk', 'wb') as f:
            pickle.dump(label_dictionary, f)
        output_dictionary = {i: labels for i, labels in enumerate(labels)}

        self.vocab_size = len(word_dictionary.keys())  # 词汇表大小
        print('vocab size', self.vocab_size)

        label_size = len(label_dictionary.keys())  # 标签类别数量
        print('label size', label_size)
        # 序列填充，按input_shape填充，长度不足的按0补充
        X = [[word_dictionary[word] for word in sent] for sent in X]
        self.X = sequence.pad_sequences(maxlen=300, sequences=X, padding='post', value=0)
        y = [[label_dictionary[sent]] for sent in y]
        y = [np_utils.to_categorical(label, num_classes=label_size) for label in y]
        self.y = np.array([list(_[0]) for _ in y])
        print('X shape', self.X.shape)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                                                            self.X, self.y,
                                                            test_size=0.1,
                                                            random_state=22)
        #########################################################################
        # 相应的word2vect的模型
        # self.data['r_result'] = self.data.star.apply(uc.snow_result)
        # self.data['cut_comment'] = self.data.comment.apply(us.svm_chinese_word_cut)
        # X = self.data['cut_comment']
        # # print(X[0])#每个评论的分词结果形如:['w1', 'w2', 'w3', 'w4']
        # # 相应的word2vect的模型
        # y = self.data.r_result
        # self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=22)
        #
        # train_model = us.model(self.X_train, '../model/svm_train.model')
        # test_model = us.model(self.X_test, '../model/svm_test.model')
        # # 相应的词向量
        # self.train_vect, self.test_vect = us.get_train_test_vec(self.X_train, self.X_test, train_model, test_model)
        # print('train vect shape', self.train_vect.shape)
        # print('train vect', self.train_vect)
        ##########################################################################

    def lstm(self):

        ###########################################
        # layers.Embedding(input_dim=,
        #                 output_dim=,
        #                 embeddings_initializer='uniform',
        #                 embeddings_regularizer=None,
        #                 activity_regularizer=None,
        #                 embeddings_constraint=None,
        #                 mask_zero=False,
        #                 input_length=None)
        ###########################################
        self.model = Sequential()

        embed_dim = 20
        lstm_out = 200
        #self.vocab_size = 6495
        self.model.add(Embedding(input_dim=self.vocab_size+1, output_dim=embed_dim, input_length=300, mask_zero=True))
        # self.model.add(Embedding(input_dim=301, output_dim=embed_dim, input_length=300, mask_zero=True))
        self.model.add(LSTM(lstm_out, input_shape=(self.X.shape[0], self.X.shape[1])))
        # self.model.add(LSTM(lstm_out, input_shape=(self.train_vect.shape[0], self.train_vect.shape[1])))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(units=2, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy',
                        optimizer='adam',
                        metrics=['accuracy'])
        # plot_model(self.model, to_file='../model/model_lstm.png', show_shapes=True)
        print(self.model.summary())
    #调整超参数，epoch, 数据集
    def train(self):

        # Y = pd.get_dummies(self.y_train).values
        # Y_test = pd.get_dummies(self.y_test).values

        # print('X test shape', self.X_test.shape)
        # print('Y test shape', self.y_test.shape)
        # print('lstm train start')
        # print('X train shape', self.X_train.shape)
        # print('Y train shape', self.y_train.shape)
        batch_size = 32
        print('##########################lstm start to evaluate##########################')
        self.model.fit(self.X_train, self.y_train, batch_size=batch_size, epochs=1, verbose=1)
        # self.model.fit(self.train_vect, Y, batch_size=batch_size, epochs=5, verbose=1)
        # self.model.train_on_batch(x_batch, y_batch)
        #评估
        # loss_and_metrics = self.model.evaluate(self.X_test, self.y_test, batch_size=batch_size)
        # print('loss', loss_and_metrics)
        # self.model.save('../model/lstm.h5')
        #预测
        # classes = model.predict(x_test, batch_size=128)




def main():
    data = '../data/comments.csv'
    # draw(data)
    lstm = LSTM_U(data)
    lstm.pro_precess()
    lstm.lstm()
    lstm.train()


def draw(data):

        font = font_manager.FontProperties(fname='yahei.ttf')
        data = pd.read_csv(data)
        data['length'] = data['comment'].apply(lambda x: len(x))
        len_df = data.groupby('length').count()

        # 句子长度
        sen_len = len_df.index.tolist()

        # 句子长度出现频率
        sen_freq = len_df['comment'].tolist()

        # print(sen_len)
        # print(sen_freq)
        # print(len_df)
        plt.bar(sen_len, sen_freq)
        plt.title("句子长度及出现频数统计图", fontproperties=font)
        plt.xlabel("句子长度", fontproperties=font)
        plt.ylabel("句子长度出现的频数", fontproperties=font)
        plt.savefig("句子长度及出现频数统计图.png")
        plt.close()
        # 绘制句子长度累积分布函数(CDF)
        sent_pentage_list = [(count / sum(sen_freq)) for count in accumulate(sen_freq)]

        # 绘制CDF
        plt.plot(sen_len, sent_pentage_list)

        # 寻找分位点为quantile的句子长度
        quantile = 0.91
        # print(list(sent_pentage_list))
        for length, per in zip(sen_len, sent_pentage_list):
            if round(per, 2) == quantile:
                index = length
                break
        print("\n分位点为%s的句子长度:%d." % (quantile, index))

        # 绘制句子长度累积分布函数图
        plt.plot(sen_len, sent_pentage_list)
        plt.hlines(quantile, 0, index, colors="c", linestyles="dashed")
        plt.vlines(index, 0, quantile, colors="c", linestyles="dashed")
        plt.text(0, quantile, str(quantile))
        plt.text(index, 0, str(index))
        plt.title("句子长度累积分布函数图", fontproperties=font)
        plt.xlabel("句子长度", fontproperties=font)
        plt.ylabel("句子长度累积频率", fontproperties=font)
        plt.savefig("句子长度累积分布函数图.png")
        plt.close()
if __name__ == '__main__':

    main()
