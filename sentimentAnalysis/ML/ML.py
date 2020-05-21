# coding=utf-8

# Change by Cz.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

import sklearn
from sklearn.preprocessing import  StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import plot_importance

from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.ensemble import  RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import tree, svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis


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
        self.data = self.data[(self.data['star'] == 50) | (self.data['star'] == 10)]
        # X = self.data['comment']
        # print(X)
        # X = [i for i in self.data['comment'] if self.data['star'] == 50 or self.data['star'] == 10].toarray()
        # print(X.shape)
        # self.data['sentiment'] = ['positive' if (x == 50) else 'negative' for x in self.data['star']]
        # y = self.data['sentiment']
        self.data['cut_comment'] = self.data.comment.apply(uc.chinese_word_cut)
        X = self.data['cut_comment']
        print('X shape', X.shape)
        print(self.data.head(10))
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
        plt.savefig(model_name+'train number-size.png')
        plt.show()


    def train(self, model_name):
        print(model_name+ '  start to train')
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
        print("**********************评测数据**********************")
        # AUC
        auc_scores = cross_val_score(self.clf, X_train, y_train, cv=5, scoring='roc_auc')
        # Accuracy
        accuracy_scores = cross_val_score(self.clf, X_train, y_train, cv=5, scoring='accuracy')
        #f1
        # f1_scores = cross_val_score(self.clf, X_train, y_train, cv=5, scoring='f1')
        print("(AUC/Accuracy)*************%0.4f/%0.4f" % (auc_scores.mean(),
                                                            accuracy_scores.mean()))
        # print("F1 Score*************%0.4f" % f1_scores.mean())

        self.diagram(model_name, 'accuracy', X_train, y_train)

    def test(self, model_name):
        print(model_name+' start to test')
        data = pd.read_csv('../data/Test_TF_IDF.csv',
                           encoding='utf-8')
        # data.info()
        # data.describe()
        # 数据和标签分开存放
        exc_cols = [u'class']
        cols = [c for c in data.columns if c not in exc_cols]
        X_test = data.loc[:, cols]
        y_test = data['class'].values
        self.clf.fit(X_test, y_test)

        nb_result = self.clf.predict(X_test)
        # print(nb_result)
        print("**********************评测数据**********************")
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
    # clf = LogisticRegression(penalty='l2', solver='lbfgs')
    # clf = KNeighborsClassifier()
    # clf = DecisionTreeClassifier()
    # clf = GradientBoostingClassifier(n_estimators=30)
    # clf = LinearDiscriminantAnalysis()#线性判别分析
    # clf = QuadraticDiscriminantAnalysis() #二次判别分析
    # clf = MultinomialNB()
    # clf = GaussianNB()
    # clf = svm.SVC(probability=True, C=50, kernel='rbf', verbose=False, gamma='scale')
    # clf = svm.NuSVC(gamma='scale')
    # clf = RandomForestClassifier(n_estimators=30)
    # clf = AdaBoostClassifier(n_estimators=30)
    clf = XGBClassifier(learning_rate=0.01,
                        n_estimators=30,#树的个数
                        max_depth=4,#树的深度
                        min_child_weight=1,  # 叶子节点最小权重
                        gamma=0.,  # 惩罚项中叶子结点个数前的参数
                        subsample=1,  # 所有样本建立决策树
                        colsample_btree=1,  # 所有特征建立决策树
                        scale_pos_weight=1,  # 解决样本个数不平衡的问题
                        random_state=27,  # 随机数
                        slient=0
                        )
    model = ML(data, clf)
    model.pro_process()
    model.train('XGBClassifier')
    model.test('XGBClassifier')
    model.save_model('../model/XGBClassifier.pkl')


if __name__ == '__main__':
    main()
#SVC

#LogisticRegression

#KNeighborsClassifier

#DecisionTreeClassifier
# Accuracy
# 0.8971848225214198
# Precision
# 0.9138386587714091
# Recall
# 0.868078626799557
# F1 score
# 0.8838309973593337
#MultinomialNB
# Accuracy
# 0.7862097103223175
# Precision
# 0.8096797106061899
# Recall
# 0.7283130306386121
# F1 score
# 0.7425853623339453
# saved done

# GaussianNB
# Accuracy
# 0.7013463892288861
# Precision
# 0.7600195694716243
# Recall
# 0.7566445182724253
# F1 score
# 0.7013101431208496

#NuSVC
# Accuracy
# 0.8890248878008976
# Precision
# 0.9090896778610089
# Recall
# 0.8565430047988187
# F1 score
# 0.8736496413447412
# saved done


#RandomForestClassifier_30
# Accuracy
# 0.8963688290493677
# Precision
# 0.9154224854378397
# Recall
# 0.8658176448874123
# F1 score
# 0.8824649521064984

#AdaBoostClassifier_30
# Accuracy
# 0.7164422684618523
# Precision
# 0.7757349114889349
# Recall
# 0.6250922849760059
# F1 score
# 0.6158987211618789