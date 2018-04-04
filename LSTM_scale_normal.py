# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

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
from keras.layers import LSTM

np.random.seed(0)

if __name__ == "__main__":
    SPLIT_SIZE = 0.2
    NUM_LABEL = 4

    # Read review text data.
    reviews = []
    with open("./scaledata/Steve+Rhodes/subj.Steve+Rhodes") as f:
        for line in f:
            reviews.append(line)

    print(len(reviews))

    # Read corresponding labels.
    labels = []
    with open("./scaledata/Steve+Rhodes/label.4class.Steve+Rhodes") as f:
        for line in f:
            if line is None:
                continue
            labels.append(int(line.strip('\n')))

    print(len(labels))

    # Vectorize label
    labels = np_utils.to_categorical(labels, NUM_LABEL)

    # text pre-processing
    stop_words = set(stopwords.words('english'))
    stop_words.update(['.', ',', '"', "'", ':', ';', '(', ')', '[', ']', '{', '}'])
    stemmer = SnowballStemmer("english")

    print
    "pre-processing train docs..."
    processed_docs_train = []
    for doc in reviews:
        tokens = word_tokenize(doc)
        filtered = [word for word in tokens if word not in stop_words]
        stemmed = [stemmer.stem(word) for word in filtered]
        processed_docs_train.append(stemmed)

    # The frequency of different words, the len of dict is the len of the vector to represent each phrase
    dictionary = corpora.Dictionary(processed_docs_train)
    dictionary_size = len(dictionary.keys())

    print "dictionary size: ", dictionary_size
    # dictionary.save('dictionary.dict')
    # corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

    print "converting to token ids..."
    word_id_train, word_id_len = [], []
    for doc in processed_docs_train:
        word_ids = [dictionary.token2id[word] for word in doc]
        word_id_train.append(word_ids)
        word_id_len.append(len(word_ids))

    word_id_train = sequence.pad_sequences(word_id_train)

    # Split the dataset
    doc_terms_train, doc_terms_test, y_train, y_test \
        = train_test_split(word_id_train, labels, test_size=SPLIT_SIZE)

    # LSTM
    print "fitting LSTM ..."
    # model = Sequential()
    # model.add(Embedding(dictionary_size, 128))
    # model.add(LSTM(128, dropout_W=0.2, dropout_U=0.2))
    # model.add(Dense(NUM_LABEL))
    # model.add(Activation('softmax'))
    #
    # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # model.fit(doc_terms_train, y_train, nb_epoch=100, batch_size=512, verbose=1)
    #
    # # serialize model to JSON
    # model_json = model.to_json()
    # with open("./Model/LSTM_model_epoch100.json", "w") as json_file:
    #     json_file.write(model_json)
    # # serialize weights to HDF5
    # model.save_weights("./Model/LSTM_model_epoch100.h5")
    # print("Saved model to disk")

    # load json and create model
    json_file = open('./Model/LSTM_model_epoch100.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("./Model/LSTM_model_epoch100.h5")
    print("Loaded model from disk")
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


    # Calculate the accuracy of testing data.
    test_pred = model.predict_classes(doc_terms_test)
    test_pred = np_utils.to_categorical(test_pred, NUM_LABEL)
    print test_pred

    print "Accracy score:" + str(accuracy_score(test_pred, y_test))
