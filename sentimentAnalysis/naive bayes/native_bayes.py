# coding=utf-8

# Change by Cz.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import learning_curve
from sentimentAnalysis import utils as un


class NB(object):
    def __init__(self, data):
        self.data = pd.read_csv(data)
        self.nb = MultinomialNB()
        self.X_train = self.X_test = None
        self.y_train = self.y_test = None
        self.vect = self.tfidftransformer = None

    def pro_process(self):
        self.data['r_result'] = self.data.star.apply(un.snow_result)
        self.data['cut_comment'] = self.data.comment.apply(un.chinese_word_cut)
        X = self.data['cut_comment']
        y = self.data.r_result
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=22)
        stopwords = un.get_stop_words('../stopwords/hit_stopwords.txt')
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
        train_size, train_scores, test_scores = learning_curve(self.nb, X_train_vect, self.y_train,
                                                               cv=10, scoring='f1_micro',
                                                               train_sizes=np.linspace(0.1,1.0,5))
        mean_train = np.mean(train_scores, 1)  # (5,)
        # 得到得分范围的上下界
        upper_train = np.clip(mean_train + np.std(train_scores, 1), 0, 1)
        lower_train = np.clip(mean_train - np.std(train_scores, 1), 0, 1)

        mean_test = np.mean(test_scores, 1)
        # 得到得分范围的上下界
        upper_test = np.clip(mean_test + np.std(test_scores, 1), 0, 1)
        lower_test = np.clip(mean_test - np.std(test_scores, 1), 0, 1)
        plt.figure('Fig1')
        plt.plot(train_size, mean_train, 'ro-', label='train')
        plt.plot(train_size, mean_test, 'go-', label='test')
        ##填充上下界的范围
        plt.fill_between(train_size, upper_train, lower_train, alpha=0.2,  # alpha：覆盖区域的透明度[0,1],其值越大，表示越不透明
                         color='r')
        plt.fill_between(train_size, upper_test, lower_test, alpha=0.2,
                         color='g')
        plt.grid()
        plt.xlabel('train size')
        plt.ylabel('f1')
        plt.legend(loc='lower right')
        plt.title('NB')
        plt.savefig('train number-size.png')
        plt.show()
        print(train_score)

    def test(self):
        X_test_vect = self.vect.transform(self.X_test)
        print(self.nb.score(X_test_vect, self.y_test))
        nb_result = self.nb.predict(X_test_vect)
        print(nb_result)
        print('accuracy')
        print(accuracy_score(self.y_test, nb_result))
        print('precision')
        print(precision_score(self.y_test, nb_result, average='macro'))
        print('recall')
        print(recall_score(self.y_test, nb_result, average='macro'))
        print('f1 score')
        print(f1_score(self.y_test, nb_result, average='macro'))

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
    nb.save_model('../model/NB.pkl')


if __name__ == '__main__':
    main()
    # print(sorted(sklearn.metrics.SCORERS.keys()))
    # choice = [0, 1]
    # print(np.random.choice(choice, 1, [0.9, 0.1])[0])