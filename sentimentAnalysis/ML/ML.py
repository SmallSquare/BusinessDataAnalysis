# coding=utf-8

# Change by Cz.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import sklearn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import learning_curve
from sklearn.ensemble import  RandomForestClassifier, AdaBoostClassifier
from sklearn import tree, svm
from sentimentAnalysis.utils import common as uc


class ML(object):
    def __init__(self, data, clf):
        self.data = pd.read_csv(data)
        self.clf = clf
        self.vect = None
        self.tfidftransformer = None
        self.X_train = self.X_test = None
        self.y_train = self.y_test = None

    def pro_process(self):
        self.data['cut_comment'] = self.data.comment.apply(uc.chinese_word_cut)
        X = self.data['cut_comment']
        y = self.data.star
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                                                        X, y,
                                                        test_size=0.2,
                                                        random_state=22)
        stopwords = uc.get_stop_words('../stopwords/hit_stopwords.txt')
        #将文本转为n-gram词频矩阵
        self.vect = CountVectorizer(max_df=0.8,
                                    min_df=3,
                                    token_pattern=u'(?u)\\b[^\\d\\W]\\w+\\b',
                                    stop_words=frozenset(stopwords),
                                    ngram_range=(1, 3),
                                    max_features=400)
        self.tfidftransformer = TfidfTransformer()
        uc.process(self.X_train, self.y_train, self.vect, self.tfidftransformer, 'Train_TF_IDF.csv')
        print('train data process done')
        uc.process(self.X_test, self.y_test, self.vect, self.tfidftransformer, 'Test_TF_IDF.csv')
        print('test data process done')

    def diagram(self, model_name, score, x, y):
        #score: ['accuracy', 'adjusted_mutual_info_score', 'adjusted_rand_score', 'average_precision',
        # 'balanced_accuracy', 'brier_score_loss', 'completeness_score', 'explained_variance', 'f1',
        # 'f1_macro', 'f1_micro', 'f1_samples', 'f1_weighted', 'fowlkes_mallows_score',
        # 'homogeneity_score', 'jaccard', 'jaccard_macro', 'jaccard_micro', 'jaccard_samples',
        # 'jaccard_weighted', 'max_error', 'mutual_info_score', 'neg_log_loss', 'neg_mean_absolute_error',
        # 'neg_mean_squared_error', 'neg_mean_squared_log_error', 'neg_median_absolute_error',
        # 'normalized_mutual_info_score', 'precision', 'precision_macro', 'precision_micro',
        # 'precision_samples', 'precision_weighted', 'r2', 'recall', 'recall_macro', 'recall_micro',
        # 'recall_samples', 'recall_weighted', 'roc_auc', 'v_measure_score']

        train_score = self.clf.score(x, y)
        train_size, train_scores, test_scores = learning_curve(self.clf, x, y,
                                                               cv=10, scoring=score,
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
        plt.ylabel(score)
        plt.legend(loc='lower right')
        plt.title(model_name)
        plt.savefig('MultinomialNB train number-size.png')
        plt.show()


    def train(self):
        data = pd.read_csv('../data/Train_TF_IDF.csv',
                           encoding='utf-8')
        # data.info()
        # data.describe()
        # 数据和标签分开存放
        exc_cols = [u'class']
        cols = [c for c in data.columns if c not in exc_cols]
        X_train = data.loc[:, cols]
        y_train = data['class'].values

        self.clf.fit(X_train, y_train)
        # AUC
        auc_scores = cross_val_score(self.clf, X_train, y_train, cv=5, scoring='roc_auc')
        # Accuracy
        accuracy_scores = cross_val_score(self.clf, X_train, y_train, cv=5, scoring='accuracy')
        #f1
        f1_scores = cross_val_score(self.clf, X_train, y_train, cv=5, scoring='f1')
        print("(AUC/Accuracy)*************%0.4f/%0.4f" % (auc_scores.mean(),
                                                            accuracy_scores.mean()))
        print("F1 Score*************%0.4f" % f1_scores.mean())

        # self.diagram('MultinomialNB', 'accuracy', X_train, y_train)

    def test(self):
        data = pd.read_csv('../data/Test_TF_IDF.csv',
                           encoding='utf-8')
        data.info()
        data.describe()
        # 数据和标签分开存放
        exc_cols = [u'class']
        cols = [c for c in data.columns if c not in exc_cols]
        X_test = data.loc[:, cols]
        y_test = data['class'].values
        self.clf.fit(X_test, y_test)

        nb_result = self.clf.predict(X_test)
        # print(nb_result)
        print('Accuracy')
        print(accuracy_score(y_test, nb_result))
        print('Precision')
        print(precision_score(y_test, nb_result, average='macro'))
        print('Recall')
        print(recall_score(y_test, nb_result, average='macro'))
        print('F1 score')
        print(f1_score(y_test, nb_result, average='macro'))

    def save_model(self, model_export_path):

        # 朴素贝叶斯中的多项式分类器
        try:
            with open(model_export_path, 'wb') as f:
                d = {
                    "clf": self.clf,
                    "vectorizer": self.vect,
                    "tfidftransformer": self.tfidftransformer,
                }
                pickle.dump(d, f)
                print('saved done')
        except:
            f.close()


def main():
    data = '../data/comments.csv'
    clf = MultinomialNB()
    # clf = GaussianNB()
    # clf = svm.SVC(probability=True, C=50, kernel='rbf', verbose=False, gamma='scale')
    # clf = svm.NuSVC(gamma='scale')
    # clf = RandomForestClassifier(n_estimators=30)
    # clf = AdaBoostClassifier(n_estimators=30)
    model = ML(data, clf)
    model.pro_process()
    model.train()
    model.test()
    model.save_model('../model/MultinomialNB.pkl')


if __name__ == '__main__':
    main()