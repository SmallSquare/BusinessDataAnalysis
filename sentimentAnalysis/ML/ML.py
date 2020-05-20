# coding=utf-8

# Change by Cz.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import learning_curve
from sklearn.ensemble import  RandomForestClassifier, AdaBoostClassifier
from sklearn import tree, svm
from sentimentAnalysis.utils import common as uc, native_bayes_tools as un


class NB(object):
    def __init__(self, data):
        self.data = pd.read_csv(data)
        # self.nb = MultinomialNB()
        # self.nb = GaussianNB()
        self.nb = svm.SVC(probability=True, C=50, kernel='rbf', verbose=False, gamma='scale')
        # self.nb = svm.NuSVC(gamma='scale')
        # self.nb = RandomForestClassifier(n_estimators=30)
        # self.nb = AdaBoostClassifier(n_estimators=30)
        self.X_train = self.X_test = None
        self.y_train = self.y_test = None

    def pro_process(self):
        # self.data['r_result'] = self.data.star.apply(uc.snow_result)
        self.data['cut_comment'] = self.data.comment.apply(un.chinese_word_cut)
        X = self.data['cut_comment']
        y = self.data.star
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                                                        X, y,
                                                        test_size=0.2,
                                                        random_state=22)
        stopwords = un.get_stop_words('../stopwords/hit_stopwords.txt')
        #将文本转为n-gram词频矩阵
        vect = CountVectorizer(max_df=0.8,
                                    min_df=3,
                                    token_pattern=u'(?u)\\b[^\\d\\W]\\w+\\b',
                                    stop_words=frozenset(stopwords),
                                    ngram_range=(1, 3),
                                    max_features=300)
        tfidftransformer = TfidfTransformer()
        X_train_vect = vect.fit_transform(self.X_train).toarray()
        print('X_train_vect', X_train_vect)
        tfidf = tfidftransformer.fit_transform(X_train_vect).toarray()
        print('tdidf',tfidf)

        featureName = vect.get_feature_names()
        featureName = list(map(lambda vec: vec.replace(' ', '_'), featureName))
        featureName.append('class')
        labelList = self.y_train
        labelList_ = list(map(lambda label: 1 if label >= 30 else 0, labelList))
        print(labelList_)
        name = np.matrix(featureName)
        tfidf = np.c_[np.matrix(tfidf), np.array(labelList_)]
        final = np.vstack((name, tfidf))
        df = pd.DataFrame(final)
        print('df', df)
        # 最终特征名称和数据保存在csv文件中
        df.to_csv('../data/TF-IDF.csv',
                  index=False, header=False, encoding='utf-8')

    def diagram(self):
        train_score = self.nb.score(X_train_vect, self.y_train)
        train_size, train_scores, test_scores = learning_curve(self.nb, X_train_vect, self.y_train,
                                                               cv=10, scoring='f1_micro',
                                                               train_sizes=np.linspace(0.1, 1.0, 5))
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


    def train(self):
        data = pd.read_csv('../data/TF-IDF.csv',
                           encoding='utf-8')
        data.info()
        data.describe()
        # 数据和标签分开存放
        exc_cols = [u'class']
        cols = [c for c in data.columns if c not in exc_cols]
        X_train = data.loc[:, cols]
        y_train = data['class'].values

        # X_train_vect = self.vect.fit_transform(self.X_train)
        self.nb.fit(X_train, y_train)
        # train_score = self.nb.score(X_train, y_train)
        # train_size, train_scores, test_scores = learning_curve(self.nb, X_train, y_train,
        #                                                        cv=10, scoring='f1_micro',
        #                                                        train_sizes=np.linspace(0.1, 1.0, 5))
        # mean_train = np.mean(train_scores, 1)  # (5,)
        # # 得到得分范围的上下界
        # upper_train = np.clip(mean_train + np.std(train_scores, 1), 0, 1)
        # lower_train = np.clip(mean_train - np.std(train_scores, 1), 0, 1)
        #
        # mean_test = np.mean(test_scores, 1)
        # # 得到得分范围的上下界
        # upper_test = np.clip(mean_test + np.std(test_scores, 1), 0, 1)
        # lower_test = np.clip(mean_test - np.std(test_scores, 1), 0, 1)
        # plt.figure('Fig1')
        # plt.plot(train_size, mean_train, 'ro-', label='train')
        # plt.plot(train_size, mean_test, 'go-', label='test')
        # ##填充上下界的范围
        # plt.fill_between(train_size, upper_train, lower_train, alpha=0.2,  # alpha：覆盖区域的透明度[0,1],其值越大，表示越不透明
        #                  color='r')
        # plt.fill_between(train_size, upper_test, lower_test, alpha=0.2,
        #                  color='g')
        # plt.grid()
        # plt.xlabel('train size')
        # plt.ylabel('f1')
        # plt.legend(loc='lower right')
        # plt.title('NB')
        # plt.savefig('train number-size.png')
        # plt.show()
        # print(train_score)
        # AUC
        auc_scores = cross_val_score(self.nb, X_train, y_train, cv=5, scoring='roc_auc')
        # Accuracy
        accuracy_scores = cross_val_score(self.nb, X_train, y_train, cv=5, scoring='accuracy')
        print("(AUC/Accuracy)*************%0.4f/%0.4f" % (auc_scores.mean(),
                                                            accuracy_scores.mean()))
        # self.diagram()

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
    data = '../data/comments_short.csv'
    nb = NB(data)
    # nb.pro_process()
    nb.train()
    # nb.test()
    # nb.save_model('../model/NB.pkl')


if __name__ == '__main__':
    main()
    # print(sorted(sklearn.metrics.SCORERS.keys()))
    # choice = [0, 1]
    # print(np.random.choice(choice, 1, [0.9, 0.1])[0])