# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from scipy import sparse

import gensim.models.doc2vec
from gensim.test.test_doc2vec import ConcatenatedDoc2Vec
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from nltk import word_tokenize, pos_tag
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

import re
from gensim.models import doc2vec
from collections import namedtuple
from sklearn.svm import LinearSVC
#from textblob.classifiers import NaiveBayesClassifier
from collections import OrderedDict
import multiprocessing
from sklearn.svm import SVC



# train test split
SPLIT_SIZE = 0.2

# Read review text data.

def scaleData(SELECT):
    reviews = []
    with open("/Users/lizy/Downloads/Q3/Information_Retrieval/project/NLP_YSL/scaledata/"+SELECT+"/subj."+SELECT) as f:
        for line in f:
            reviews.append(line)

    print(len(reviews))

    labels = []
    with open("/Users/lizy/Downloads/Q3/Information_Retrieval/project/NLP_YSL/scaledata/"+SELECT+"/label."+str(num_of_class)+"class."+SELECT) as f:
        for line in f:
            labels.append(line)

    sentences = [];
    for line in reviews:
        new_line = re.sub(r'\d', "", line)
        sentences.append(new_line)

    print(len(labels))

    return sentences, labels


# data selection

''' ================= 1 - SCALE DATA; 2 - AMAZON ==========================='''
DATA_TYPE = 2
if DATA_TYPE == 1:
    num_of_class = 4
    NAME = ['Dennis+Schwartz', 'James+Berardinelli', 'Scott+Renshaw', 'Steve+Rhodes']
    SELECT = NAME[3]
    reviews, labels = scaleData(SELECT)
else:
    train_df = pd.read_csv('/Users/lizy/Downloads/Q3/Information_Retrieval/project/NLP_YSL-master/kaggledata/records.tsv', sep='\t', header=0)
    reviews = train_df['Review'].values
    labels = train_df['Score'].values
    num_labels = len(np.unique(labels))
    print(pd.value_counts(labels))
    print(num_labels)



stop_words = set(stopwords.words('english'))
stop_words.update(['.', ',', '"', "'", ':', ';', '(', ')', '[', ']', '{', '}'])
stemmer = SnowballStemmer("english")

print
("pre-processing train docs...")



''' ================ Doc2vec ==================='''
# Transform data (you can add more data preprocessing steps)

docs = []
analyzedDocument = namedtuple('AnalyzedDocument', 'words tags')
for i, text in enumerate(reviews):
    tokens = word_tokenize(text)
    filtered = [word for word in tokens if word not in stop_words]
    stemmed = [stemmer.stem(word) for word in filtered]
    tags = [i]
    docs.append(analyzedDocument(stemmed, tags))

cores = multiprocessing.cpu_count()
assert gensim.models.doc2vec.FAST_VERSION > -1, "This will be painfully slow otherwise"

simple_models = [
    # PV-DM w/ concatenation - window=5 (both sides) approximates paper's 10-word total window size
    doc2vec.Doc2Vec(dm=1, dm_concat=1, size=200, window=5, negative=5, hs=0, min_count=2, workers=cores),
    # PV-DBOW
    doc2vec.Doc2Vec(dm=0, size=200, negative=5, hs=0, min_count=2, workers=cores),
    # PV-DM w/ average
    doc2vec.Doc2Vec(dm=1, dm_mean=1, size=200, window=10, negative=5, hs=0, min_count=2, workers=cores),
]

# Speed up setup by sharing results of the 1st model's vocabulary scan
simple_models[0].build_vocab(docs)  # PV-DM w/ concat requires one special NULL word so it serves as template
print(simple_models[0])
for model in simple_models[1:]:
    model.reset_from(simple_models[0])
    print(model)

models_by_name = OrderedDict((str(model), model) for model in simple_models)

models_by_name['dbow+dmm'] = ConcatenatedDoc2Vec([simple_models[1], simple_models[2]])
models_by_name['dbow+dmc'] = ConcatenatedDoc2Vec([simple_models[1], simple_models[0]])

alpha, min_alpha, passes = (0.025, 0.001, 20)
alpha_delta = (alpha - min_alpha) / passes




'''================ TFidfVectorizer ================'''
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

    x_train = vectorizer.fit_transform(reviews)
    #print("feature names:")
    #print(vectorizer.get_feature_names())

    return x_train


def split_data(data):
    x_train_uni, x_test_uni, y_train, y_test \
        = train_test_split(data, labels, test_size=SPLIT_SIZE)
    return x_train_uni,x_test_uni,y_train,y_test

def ova_accuracy(data):
    x_train, x_test, y_train, y_test = split_data(data)
    print ('*************************\nSVM\n*************************')
    clf = LinearSVC(random_state=0)
    clf.fit(x_train, y_train)
    LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
              intercept_scaling=1, loss='squared_hinge', max_iter=1000,
              multi_class='ovr', penalty='l2', random_state=0, tol=0.0001,
              verbose=0)
    pred = clf.predict(x_test)
    #print(pred)
    #calculate_result(x_test_uni.target,pred);
    acc = np.mean(pred==y_test)
    return acc


# unigram
x_uni = review2vector_tfidf((1,1))

#bigram
x_bi = review2vector_tfidf((2,2))

# mix
x_mix = review2vector_tfidf((1,2))

# d2v
train_set,test_set,y_train,y_test = split_data(docs)
for name, train_model in models_by_name.items():
        # Train
    train_model.alpha, train_model.min_alpha = alpha, alpha

    train_model.train(train_set, total_examples=len(docs), epochs=1)

    vectors = [train_model.docvecs[doc.tags[0]] for doc in docs]

    sV = sparse.csr_matrix(np.asmatrix(np.asarray(vectors)))

    ova_acc_d2v = ova_accuracy(sV)
    print(name)
    print(ova_acc_d2v)

'''=============== OVA ================='''

print('=============== OVA =================')

#Unigram

ova_acc_uni = ova_accuracy(x_uni)
print("unigram:"+str(ova_acc_uni))

#Bigram

ova_acc_bi = ova_accuracy(x_bi)
print("bigram:"+str(ova_acc_bi))

# Mix

ova_acc_mix = ova_accuracy(x_mix)
print("mixgram:"+str(ova_acc_mix))



#Result

'''
Doc2Vec(dm/c,d100,n5,w5,mc2,s0.001,t4)
0.427184466019

Doc2Vec(dbow,d100,n5,mc2,s0.001,t4)
0.509708737864

Doc2Vec(dm/m,d100,n5,w10,mc2,s0.001,t4)
0.402912621359

dbow+dmm
0.480582524272

dbow+dmc
0.364077669903

unigram:0.543689320388

bigram:0.388349514563

mixgram:0.509708737864'''


'''Doc2Vec(dm/c,d100,n5,w5,mc2,s0.001,t4)
0.507633587786

Doc2Vec(dbow,d100,n5,mc2,s0.001,t4)
0.469465648855

Doc2Vec(dm/m,d100,n5,w10,mc2,s0.001,t4)
0.477099236641

dbow+dmm
0.43893129771

dbow+dmc
0.458015267176

unigram:0.629770992366

bigram:0.507633587786

mixgram:0.519083969466'''

'''Doc2Vec(dm/c,d100,n5,w5,mc2,s0.001,t4)
0.414364640884
*************************
SVM
*************************
Doc2Vec(dbow,d100,n5,mc2,s0.001,t4)
0.348066298343
*************************
SVM
*************************
Doc2Vec(dm/m,d100,n5,w10,mc2,s0.001,t4)
0.425414364641
*************************
SVM
*************************
dbow+dmm
0.381215469613
*************************
SVM
*************************
dbow+dmc
0.348066298343
=============== OVA =================
*************************
SVM
*************************
unigram:0.441988950276
*************************
SVM
*************************
bigram:0.359116022099
*************************
SVM
*************************
mixgram:0.469613259669'''

'''Doc2Vec(dm/c,d100,n5,w5,mc2,s0.001,t4)
0.435028248588
*************************
SVM
*************************
Doc2Vec(dbow,d100,n5,mc2,s0.001,t4)
0.435028248588
*************************
SVM
*************************
Doc2Vec(dm/m,d100,n5,w10,mc2,s0.001,t4)
0.440677966102
*************************
SVM
*************************
dbow+dmm
0.457627118644
*************************
SVM
*************************
dbow+dmc
0.443502824859
=============== OVA =================
*************************
SVM
*************************
unigram:0.590395480226
*************************
SVM
*************************
bigram:0.516949152542
*************************
SVM
*************************
mixgram:0.598870056497'''