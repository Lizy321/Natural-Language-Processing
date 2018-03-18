import numpy as np
import pandas as pd
import re

from gensim import corpora
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding
from keras.layers import LSTM

np.random.seed(0)

if __name__ == "__main__":
    SPLIT_SIZE = 0.3

    # load data
    train_df = pd.read_csv('./data/train.tsv', sep='\t', header=0)

    raw_docs_train = train_df['Phrase'].values
    sentiment_train = train_df['Sentiment'].values
    num_labels = len(np.unique(sentiment_train))

    # text pre-processing
    stop_words = set(stopwords.words('english'))
    stop_words.update(['.', ',', '"', "'", ':', ';', '(', ')', '[', ']', '{', '}'])
    stemmer = SnowballStemmer("english")

    print "pre-processing train docs..."
    processed_docs_train = []
    for doc in raw_docs_train:
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

    seq_len = np.round((np.mean(word_id_len) + 2 * np.std(word_id_len))).astype(int)

    # pad sequences
    word_id_train = sequence.pad_sequences(np.array(word_id_train), maxlen=seq_len)
    y_train_enc = np_utils.to_categorical(sentiment_train, num_labels)

    print word_id_train

    doc_terms_train, doc_terms_test, y_train, y_test \
        = train_test_split(word_id_train, y_train_enc, test_size=SPLIT_SIZE)

    # LSTM
    print "fitting LSTM ..."
    model = Sequential()
    model.add(Embedding(dictionary_size, 128))
    model.add(LSTM(128, dropout_W=0.2, dropout_U=0.2))
    model.add(Dense(num_labels))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(doc_terms_train, y_train, epochs=3, batch_size=256, verbose=1, validation_data=(doc_terms_test, y_test))

    # Calculate the accuracy of testing data.
    test_pred = model.predict_classes(doc_terms_test)
    test_pred = np_utils.to_categorical(test_pred, num_labels)

    print accuracy_score(test_pred, y_test)
