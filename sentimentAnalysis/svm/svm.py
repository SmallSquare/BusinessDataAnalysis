# coding=utf-8

# Change by Cz.

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sentimentAnalysis.utils import svm_tools as us, common as uc



class SVM(object):

    def __init__(self, c, data):
        self.clf = SVC(probability=True, C=c, kernel='rbf', verbose=False, gamma='scale')
        self.data = pd.read_csv(data)
        self.X_train = None
        self.y_train = None
        self.train_vect = None
        self.test_vect = None


    def pro_process(self):
        print('pro_process')
        self.data['r_result'] = self.data.star.apply(uc.snow_result)
        self.data['cut_comment'] = self.data.comment.apply(us.svm_chinese_word_cut)
        X = self.data['cut_comment']
        #print(X[0])#每个评论的分词结果形如:['w1', 'w2', 'w3', 'w4']
        y = self.data.r_result
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=22)

        #相应的word2vect的模型
        train_model = us.model(self.X_train, '../model/svm_train.model')
        test_model = us.model(self.X_test, '../model/svm_test.model')
        #相应的词向量
        self.train_vect, self.test_vect = us.get_train_test_vec(self.X_train, self.X_test, train_model, test_model)
        print('pro_process done')


    def train(self):
        print("------ SVM Classifier is training ------")

        self.clf.fit(self.train_vect, self.y_train)
        print(self.clf.score(self.train_vect, self.y_train))

        print("------ SVM Classifier training over ------")

    def test(self):
        print('test...')
        self.clf.fit(self.test_vect, self.y_test)
        print(self.clf.score(self.test_vect, self.y_test))
        print('test done...')
        svm_result = self.clf.predict(self.test_vect)
        print(svm_result)
        print('accuracy')
        print(accuracy_score(self.y_test, svm_result))
        print('precision')
        print(precision_score(self.y_test, svm_result, average='macro'))
        print('recall')
        print(recall_score(self.y_test, svm_result, average='macro'))
        print('f1 score')
        print(f1_score(self.y_test, svm_result, average='macro'))

    def save(self, path):
        with open(path, 'wb') as f:
            d = {
                'clf': self.clf,
            }
            pickle.dump(d, f)
        print('saved done')


def main():
    data = '../data/comments_short.csv'
    svm = SVM(50, data)
    svm.pro_process()
    svm.train()
    svm.test()
    svm.save('../model/svm.pkl')

if __name__ =='__main__':
    main()