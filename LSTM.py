# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from nltk import word_tokenize, pos_tag
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

from keras.utils import np_utils
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding
from keras.layers import LSTM

np.random.seed(0)


# review2vector -- transforming review to vector where num as the length of each tokens
def review2vector_tfidf(doc_terms, range):
    vectorizer = TfidfVectorizer(max_features=40000, ngram_range=range, sublinear_tf=True, binary=False,
                                 decode_error='ignore',
                                 stop_words='english')  # ngram_range=(1, 3)

    x_train = vectorizer.fit_transform(doc_terms)
    return x_train


# Unigram + pos feature
def pos_feature(new_terms_train):
    pos_review = []
    for review in new_terms_train:
        tokens = word_tokenize(review)
        tags = dict(pos_tag(tokens)).values()
        pos = ' '.join(list(tags))
        pos_review.append(review + ' ' + pos)
    return pos_review


def POSreview2vector_tfidf(pos_train, pos_test, range):
    vectorizer = TfidfVectorizer(max_features=40000, ngram_range=range, sublinear_tf=True, binary=False,
                                 decode_error='ignore', \
                                 stop_words='english')  # ngram_range=(1, 3)

    x_train = vectorizer.fit_transform(pos_train)
    x_test = vectorizer.fit_transform(pos_test)

    print("feature names:")
    print(vectorizer.get_feature_names())

    return x_train, x_test


def batch_generator(X, y, samples_per_epoch, batch_size):
    number_of_batches = samples_per_epoch / batch_size
    counter = 0
    shuffle_index = np.arange(np.shape(y)[0])
    np.random.shuffle(shuffle_index)
    X = X[shuffle_index, :]
    y = y[shuffle_index]
    while 1:
        index_batch = shuffle_index[batch_size * counter:batch_size * (counter + 1)]
        X_batch = X[index_batch, :].todense()
        y_batch = y[index_batch]
        counter += 1
        yield (np.array(X_batch), y_batch)
        if (counter < number_of_batches):
            np.random.shuffle(shuffle_index)
            counter = 0


if __name__ == "__main__":
    SPLIT_SIZE = 0.3
    NUM_LABEL = 5

    # load data
    train_df = pd.read_csv('./kaggledata/records_5554.tsv', sep='\t', header=0)

    print len(train_df)
    raw_docs_train = train_df['Review'].values
    sentiment_train = train_df['Score'].values
    num_labels = len(np.unique(sentiment_train))

    # Vectorize label
    labels = np_utils.to_categorical(sentiment_train, NUM_LABEL)
    # Unigram, Bigram, Mix, unigram+pos feature
    x_uni = review2vector_tfidf(raw_docs_train, (1, 1))

    # x_train_bi, x_train_bi = review2vector_tfidf((2, 2))
    # x_train_mix, x_train_mix = review2vector_tfidf((1, 2))
    # pos_train = pos_feature(doc_terms_train)
    # pos_test = pos_feature(doc_terms_test)
    # pos_x_train, pos_x_test = POSreview2vector_tfidf((1, 1))

    # Split the dataset
    doc_terms_train, doc_terms_test, y_train, y_test \
        = train_test_split(x_uni, labels, test_size=SPLIT_SIZE)
    print "The amount of training data: " + str(doc_terms_train.shape[0])
    print "Input vector size: " + str(doc_terms_train.shape[1])
    # LSTM
    print "fitting LSTM ..."
    model = Sequential()
    model.add(Embedding(doc_terms_train.shape[1], 128))
    model.add(LSTM(128, dropout_W=0.2, dropout_U=0.2))
    model.add(Dense(NUM_LABEL))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(np.array(doc_terms_train.todense()), y_train, nb_epoch=1, batch_size=32, verbose=1)
    # model.fit_generator(generator=batch_generator(doc_terms_train, y_train, len(train_df), 32),
    #                     nb_epoch=1, samples_per_epoch=doc_terms_train.shape[0], verbose=1)
    # # serialize model to JSON
    # model_json = model.to_json()
    # with open("./Model/LSTM_model_epoch1.json", "w") as json_file:
    #     json_file.write(model_json)
    # # serialize weights to HDF5
    # model.save_weights("./Model/LSTM_model_epoch1.h5")
    # print("Saved model to disk")

    # # load json and create model
    # json_file = open('./Model/LSTM_kaggle_model_epoch3.json', 'r')
    # loaded_model_json = json_file.read()
    # json_file.close()
    # loaded_model = model_from_json(loaded_model_json)
    # # load weights into new model
    # loaded_model.load_weights("./Model/LSTM_kaggle_model_epoch3.h5")
    # print("Loaded model from disk")
    #
    # loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Calculate the accuracy of testing data.
    test_pred = model.predict_classes(np.array(doc_terms_test.todense()))
    test_pred = np_utils.to_categorical(test_pred, NUM_LABEL)

    print "Accracy score:" + str(accuracy_score(test_pred, y_test))
