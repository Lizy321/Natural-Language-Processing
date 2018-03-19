# -*- coding: utf-8 -*-
import scipy as sp
import numpy as np
from sklearn.datasets import load_files
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from nltk import word_tokenize, pos_tag
from pandas import DataFrame
import re
import nltk
#nltk.download()

import csv
from textblob.classifiers import NaiveBayesClassifier
from nltk import NaiveBayesClassifier, classify
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
import numpy as np


'''''加载数据集，切分数据集80%训练，20%测试'''
movie_reviews = DataFrame.from_csv("/Users/lizy/Downloads/Q3/Information_Retrieval/project/data/train.tsv", sep="\t")

reviews = []
with open("/Users/lizy/Downloads/Q3/Information_Retrieval/project/NLP_YSL/scaledata" + "/" + reviewFileName) as f:
    for line in f:
        reviews.append(line)

#remove numbers
sentences = [];
for line in movie_reviews.Phrase:
    new_line = re.sub(r'\d', "", line)
    sentences.append(new_line)

#movie_reviews = load_files('endata')
doc_terms_train, doc_terms_test, y_train, y_test \
    = train_test_split(sentences, movie_reviews.Sentiment, test_size=0.3)

'''''BOOL型特征下的向量空间模型，注意，测试样本调用的是transform接口'''
# unigram
count_vec = TfidfVectorizer(binary=False, decode_error='ignore', \
                            stop_words='english')

#bigram

#print(vectorizer.get_feature_names())

# review2vector -- transforming review to vector where num as the length of each tokens
def review2vector_tfidf(range):
    vectorizer = TfidfVectorizer(max_features=40000, ngram_range=range , sublinear_tf=True, binary=False, decode_error='ignore', \
                            stop_words='english')#ngram_range=(1, 3)

    x_train = vectorizer.fit_transform(doc_terms_train)
    x_test = vectorizer.fit_transform(doc_terms_test)

    print("feature names:")
    print(vectorizer.get_feature_names())

    return x_train,x_test

x_train_uni, x_test_uni = review2vector_tfidf((1,1))
x_train_bi, x_train_bi = review2vector_tfidf((2,2))

x_train_mix, x_train_mix = review2vector_tfidf((1,2))




# POS feature


def pos_feature(new_terms_train):
    pos_review = []
    for review in new_terms_train:
        tokens = word_tokenize(review)
        tags = dict(pos_tag(tokens)).values()
        pos = ' '.join(list(tags))
        pos_review.append(review + ' ' + pos)
    return pos_review

pos_train = pos_feature(doc_terms_train)
pos_test = pos_feature(doc_terms_test)




def POSreview2vector_tfidf(range):
    vectorizer = TfidfVectorizer(max_features=40000, ngram_range=range , sublinear_tf=True, binary=False, decode_error='ignore', \
                            stop_words='english')#ngram_range=(1, 3)

    x_train = vectorizer.fit_transform(pos_train)
    x_test = vectorizer.fit_transform(pos_test)

    print("feature names:")
    print(vectorizer.get_feature_names())

    return x_train,x_test


pos_x_train, pos_x_test= POSreview2vector_tfidf((1,1))
