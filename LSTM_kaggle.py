import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from gensim import corpora
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import model_from_json
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding
from keras.layers import LSTM

np.random.seed(0)

if __name__ == "__main__":
    SPLIT_SIZE = 0.3

    # load data
    train_df = pd.read_csv('./kaggledata/train.tsv', sep='\t', header=0)

    raw_docs_train = train_df['Phrase'].values
    sentiment_train = train_df['Sentiment'].values
    num_labels = len(np.unique(sentiment_train))
    y_train_enc = np_utils.to_categorical(sentiment_train, num_labels)
    print pd.value_counts(sentiment_train)
    print num_labels
    print "After categorical label amounts: " + str(len(np.unique(y_train_enc, axis=0)))

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
    print seq_len
    word_id_train = sequence.pad_sequences(np.array(word_id_train), maxlen=seq_len, padding='pre')

    print word_id_train[0]
    print word_id_train[len(word_id_train) - 1]
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
    history = model.fit(doc_terms_train, y_train, validation_split=0.33, nb_epoch=2, batch_size=64, verbose=1)

    # Calculate the accuracy of testing data.
    # list all data in history
    print(history.history.keys())
    print(history.history['acc'])
    print(history.history['val_acc'])
    print(history.history['loss'])
    print(history.history['val_loss'])
    # summarize history for accuracy
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['training', 'validation'], loc='upper left')
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.savefig('./illustration/kaggle_acc.png')
    plt.clf()
    
    # summarize history for loss
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['training', 'validation'], loc='upper left')
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.savefig('./illustration/kaggle_loss.png')

    test_pred = model.predict_classes(doc_terms_test, batch_size=64)
    print test_pred
    test_pred = np_utils.to_categorical(test_pred, num_labels)


    print "Accracy score:" + str(accuracy_score(test_pred, y_test))
    print "Confusion Matrix: "
    print confusion_matrix(test_pred.argmax(axis=1), y_test.argmax(axis=1))
