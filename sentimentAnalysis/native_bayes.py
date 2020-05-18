# coding=utf-8

# Change by Cz.

from snownlp import SnowNLP
import numpy as np
import pandas as pd
import pickle
import jieba
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import utils.native_bayes_tools as un


class NB(object):
    def __init__(self, data):
        self.data = pd.read_csv(data)
        self.nb = MultinomialNB()
        self.X_train = self.X_test = None
        self.y_train = self.y_test = None
        self.vect = self.tfidftransformer = None

    def pro_process(self):
        self.data['snlp_result'] = self.data.comments.apply(un.snow_result)
        self.data['cut_comment'] = self.data.comments.apply(un.chinese_word_cut)
        X = self.data['cut_comment']
        y = self.data.snlp_result
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=22)
        stopwords = un.get_stop_words('hit_stopwords.txt')
        self.vect = CountVectorizer(max_df=0.8,
                                    min_df=3,
                                    token_pattern=u'(?u)\\b[^\\d\\W]\\w+\\b',
                                    stop_words=frozenset(stopwords))
        self.tfidftransformer = TfidfTransformer()
        # data.head()

    def train(self):
        X_train_vect = self.vect.fit_transform(self.X_train)
        self.nb.fit(X_train_vect, self.y_train)
        train_score = self.nb.score(X_train_vect, self.y_train)
        print(train_score)

    def test(self):
        X_test_vect = self.vect.transform(self.X_test)
        print(self.nb.score(X_test_vect, self.y_test))
        #
        # X_vec = vect.transform(X)
        # nb_result = nb.predict(X_vec)
        # data['nb_result'] = nb_result

    def save_model(self, model_export_path):
        tfidf = self.tfidftransformer.fit_transform(self.vect.fit_transform(self.X_train))
        # 朴素贝叶斯中的多项式分类器
        clf = self.nb.fit(tfidf, self.y_train)
        try:

            with open(model_export_path, 'wb') as f:
                d = {
                    "clf": clf,
                    "vectorizer": self.vect,
                    "tfidftransformer": self.tfidftransformer,
                }
                pickle.dump(d, f)
                print('saved done')
        except:
            f.close()


def main():
    data = '../data/comments.csv'
    nb = NB(data)
    nb.pro_process()
    nb.train()
    nb.test()
    nb.save_model('./model/NB.pkl')


if __name__ == '__main__':
    main()
