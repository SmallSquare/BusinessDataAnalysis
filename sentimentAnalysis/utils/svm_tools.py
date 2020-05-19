# coding=utf-8

# Change by Cz.
from gensim.models import word2vec
import jieba
import numpy as np
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