# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from scipy import sparse

from gensim import corpora
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from nltk import word_tokenize, pos_tag
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

from keras.utils import np_utils
from keras.models import model_from_json
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.layers import Dense, Activation, Embedding
from gensim.models import doc2vec
from collections import namedtuple


SPLIT_SIZE = 0.2

# Read review text data.
reviews = []
with open("/Users/lizy/Downloads/Q3/Information_Retrieval/project/NLP_YSL-master/scaledata/Steve+Rhodes/subj.Steve+Rhodes") as f:
    for line in f:
        reviews.append(line)

print(len(reviews))

# Read corresponding labels.
labels = []
with open("/Users/lizy/Downloads/Q3/Information_Retrieval/project/NLP_YSL-master/scaledata/Steve+Rhodes/label.4class.Steve+Rhodes") as f:
    for line in f:
        if line is None:
            continue
        labels.append(int(line.strip('\n')))

print(len(labels))

stop_words = set(stopwords.words('english'))
stop_words.update(['.', ',', '"', "'", ':', ';', '(', ')', '[', ']', '{', '}'])
stemmer = SnowballStemmer("english")

print
("pre-processing train docs...")


# Transform data (you can add more data preprocessing steps)

docs = []
analyzedDocument = namedtuple('AnalyzedDocument', 'words tags')
for i, text in enumerate(reviews):
    tokens = word_tokenize(text)
    filtered = [word for word in tokens if word not in stop_words]
    stemmed = [stemmer.stem(word) for word in filtered]
    tags = [i]
    docs.append(analyzedDocument(stemmed, tags))

model = doc2vec.Doc2Vec(docs, vector_size=100, window=300, min_count=1, workers=4)

vectors = []
for i in range(len(docs)):
    vectors.append(list(model.docvecs[i]))

sV = sparse.csr_matrix(np.asmatrix(np.asarray(vectors)))

doc_terms_train, doc_terms_test, y_train, y_test \
    = train_test_split(sV, labels, test_size=SPLIT_SIZE)

