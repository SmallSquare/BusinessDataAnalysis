# coding=utf-8

# code by CZ.
import  csv
from csv import  DictWriter
import pandas  as pd
import numpy as np
import jieba
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from gensim.models import word2vec

def get_stop_words(stopwords):
    with open(stopwords, encoding='utf-8') as f:
        stopwords = f.read()
    stopwords_list = stopwords.split('\n')
    stopwords = [i for i in stopwords_list]
    return stopwords

def chinese_word_cut(mytext):
    return " ".join(jieba.cut(mytext))

def so(sentence):
    seg_list = jieba.cut(sentence, cut_all=False)
    stopwords = get_stop_words('../stopwords/hit_stopwords.txt')
    temp = " ".join(seg_list).split()
    out_sentences = []
    for w in temp:
        if w not in stopwords:
            out_sentences.append(w)
    print(out_sentences)
    return out_sentences


def process(data, data_label, vect, tfidftransformer, pathname):
    data_vect = vect.fit_transform(data).toarray()
    data_tfidf = tfidftransformer.fit_transform(data_vect).toarray()

    featureName = vect.get_feature_names()
    featureName = list(map(lambda vec: vec.replace(' ', '_'), featureName))
    featureName.append('class')
    labelList = data_label
    labelList_ = list(map(lambda label: 1 if label >= 30 else 0, labelList))
    # print(labelList_)
    name = np.matrix(featureName)
    tfidf = np.c_[np.matrix(data_tfidf), np.array(labelList_)]
    final = np.vstack((name, tfidf))
    df = pd.DataFrame(final)
    # print('df', df)
    # 最终特征名称和数据保存在csv文件中
    df.to_csv('../data/'+pathname,
              index=False, header=False, encoding='utf-8')

def process_x(data, vect, tfidftransformer):
    data_vect = vect.transform(data).toarray()
    data_tfidf = tfidftransformer.transform(data_vect).toarray()

    featureName = vect.get_feature_names()
    featureName = list(map(lambda vec: vec.replace(' ', '_'), featureName))
    name = np.matrix(featureName)
    tfidf = np.matrix(data_tfidf)
    final = np.vstack((name, tfidf))
    df = pd.DataFrame(final)
    df.to_csv('../data/x.csv',
              index=False, header=False, encoding='utf-8')

def dict_of_list_to_csv(list, out_file, attributes):
    """
    :param list:   [{'title': '真心半解', 'rate': '8.0', 'id': 33420285}, {'title': '利刃出鞘', 'rate': '8.2', 'id': 30318116}]
    :param out_file: file name
    :return: csv file
    """
    out = []
    # out_id = set()
    for i in list:
        item = []
        for j in attributes:
            item.append(i[j])
        out.append(item)
        # out_id.add(i['id'])
    res = pd.DataFrame(out)
    # out_id = pd.DataFrame(out_id)
    res.to_csv('../data/'+out_file, encoding = 'utf-8', mode='a', index = False,header = None)
    # out_id.to_csv('../data/movie_id.csv', encoding='utf-8', index=False, header=['id'])

def sec_dict_of_list_to_csv(list, out_file, attributes):
    """
    :param list:
    # data = [
    #     {'key_1': {'calc1': 42, 'calc2': 3.142}},
    #     {'key_2': {'calc1': 123.4, 'calc2': 1.414}},
    #     {'key_3': {'calc1': 2.718, 'calc2': 0.577}}
    # ]
    :return:
    """
    try:
        with open('../data/'+out_file, 'wb') as f:
            writer = DictWriter(f, attributes)
            writer.writerow(dict(zip(writer.fieldnames, writer.fieldnames)))
            for i in data:
                key, values = i.items()[0]
                writer.writerow(dict(key=key, **values))
    except:
        f.close()

def list_to_csv(list, out_file):
    """
    :param list:  ['aa', 'bbb']
    :param out_file: file_name
    :return: csv file
    """
    res = pd.DataFrame(list)
    res.to_csv('../data/'+out_file, encoding = 'utf-8', mode='a', index = False,header = ['comments'])



def svm_chinese_word_cut(sentence):
    print(sentence)
    stopwordsFile = '../stopwords/hit_stopwords.txt'
    stopwords = [line.strip() for line in
                 open(stopwordsFile, encoding='utf-8').readlines()]
    sentences = []
    seg_list = jieba.cut(sentence, cut_all=False)
    # print(seg_list)
    sentences.append(" ".join(seg_list).split())
    out_sentences = []
    for obj in sentences:
        for w in obj:
            if w not in stopwords:
                out_sentences.append(w)
    print(out_sentences)
    return out_sentences

def snow_result(txt):
    if txt > 30:
        return 'positive'
    elif txt == 30:
        return 'neutral'
    else:
        return 'negative'


def model(data, model_path):
    """
    this func used to get a model of word vector
    """
    model = word2vec.Word2Vec(data, sg=0, hs=1, min_count=2, window=5, size=300)
    # word2vec.Word2Vec(self.X_train, min_count=1, size=200, window=5, negative=3, sample=0.001,
                      # sg=1, hs=1)
    # self.model = model
    # # gensim.models.Word2Vec.load()
    # print(model.wv.index2word)
    model.save(model_path)
    return model



#todo 待改进
def get_sent_vec(size, sent, model):
    """
    this func used to compute word vector
    strategy used is computer mean of word vector
    :param sent word segment of every comment
    """
    vec = np.zeros(size).reshape(1,size)
    count = 0
    for word in sent:
        try:
            vec += model[word].reshape(1,size)
            count += 1
        except:
            continue
    if count != 0:
        vec /= count
    return vec


def get_train_test_vec(x_train,x_test,train_model,test_model):
    """
    this func used to get word vector of the train and test data
    :param word segment of all commnets [['w1', 'w2', 'w3', ...], [], [], []]
    """
    train_vec = np.concatenate([get_sent_vec(300, sent, train_model) for sent in x_train])
    test_vec = np.concatenate([get_sent_vec(300, sent, test_model) for sent in x_test])
    # 保存数据
    # np.save(path+'train_vec.npy', train_vec)
    # np.save(path+'test_vec.npy', test_vec)
    return train_vec, test_vec

def get_vect(x, x_model):
    """
    this func also used to get word vector of data x
    """
    vec = np.concatenate([get_sent_vec(300, sent, x_model) for sent in x])
    return vec

def svm_chinese_word_cut(sentence):
    # print(sentence)
    stopwordsFile = '../stopwords/hit_stopwords.txt'
    stopwords = [line.strip() for line in
                 open(stopwordsFile, encoding='utf-8').readlines()]
    sentences = []
    seg_list = jieba.cut(sentence, cut_all=False)
    # print(seg_list)
    sentences.append(" ".join(seg_list).split())
    out_sentences = []
    for obj in sentences:
        for w in obj:
            if w not in stopwords:
                out_sentences.append(w)
    # print(out_sentences)
    return out_sentences