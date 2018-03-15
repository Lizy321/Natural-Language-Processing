# -*- coding: utf-8 -*-
import scipy as sp
import numpy as np
from sklearn.datasets import load_files
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from pandas import DataFrame
import re

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

#remove numbers
sentences = [];
for line in movie_reviews.Phrase:
    new_line = re.sub(r'\d', "", line)
    sentences.append(new_line)

#movie_reviews = load_files('endata')
doc_terms_train, doc_terms_test, y_train, y_test \
    = train_test_split(sentences, movie_reviews.Sentiment, test_size=0.3)

'''''BOOL型特征下的向量空间模型，注意，测试样本调用的是transform接口'''
count_vec = TfidfVectorizer(binary=False, decode_error='ignore', \
                            stop_words='english')

# training set
x_train = count_vec.fit_transform(doc_terms_train)
# LABEL --> y_train

#testing set
x_test = count_vec.transform(doc_terms_test)
#Label --> y_test


x = count_vec.transform(sentences)
y = movie_reviews.Sentiment
print(doc_terms_train)
print(count_vec.get_feature_names())
print(x_train.toarray())
print(movie_reviews.target)

#text_clf = x_train.fit(OneVsOneClassifier(LinearSVC()),np.asarray())