import numpy as np
import pandas as pd

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
from keras.layers import SimpleRNN
from gensim.models import doc2vec
from collections import namedtuple

np.random.seed(0)

if __name__ == "__main__":
    SPLIT_SIZE = 0.3
    VECTOR_SIZE = 100

    # load data
    train_df = pd.read_csv('./kaggledata/records.tsv', sep='\t', header=0)

    raw_docs_train = train_df['Review'].values
    sentiment_train = train_df['Score'].values
    num_labels = len(np.unique(sentiment_train))
    print pd.value_counts(sentiment_train)
    print sentiment_train
    print "Label's categories amount: " + str(num_labels)

    # text pre-processing
    stop_words = set(stopwords.words('english'))
    stop_words.update(['.', ',', '"', "'", ':', ';', '(', ')', '[', ']', '{', '}'])
    stemmer = SnowballStemmer("english")

    print "pre-processing train docs..."
    processed_docs_train = []
    for doc in raw_docs_train:
        tokens = word_tokenize(doc.lower())
        filtered = [word for word in tokens if word not in stop_words]
        stemmed = [stemmer.stem(word) for word in filtered]
        processed_docs_train.append(filtered)

    docs = []
    analyzedDocument = namedtuple('AnalyzedDocument', 'words tags')
    for i, text in enumerate(processed_docs_train):
        # words = text.lower().split()
        tags = [i]
        docs.append(analyzedDocument(text, tags))
    # Train model (set min_count = 1, if you want the model to work with the provided example data set)

    model_vec = doc2vec.Doc2Vec(docs, vector_size=VECTOR_SIZE, window=300, min_count=1, workers=4)
    word_id_train = model_vec.docvecs

    # # The frequency of different words, the len of dict is the len of the vector to represent each phrase
    # dictionary = corpora.Dictionary(processed_docs_train)
    # dictionary_size = len(dictionary.keys())
    # print "dictionary size: ", dictionary_size
    # # dictionary.save('dictionary.dict')
    # # corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
    #
    # print "converting to token ids..."
    # word_id_train, word_id_len = [], []
    # for doc in processed_docs_train:
    #     word_ids = [dictionary.token2id[word] for word in doc]
    #     word_id_train.append(word_ids)
    #     word_id_len.append(len(word_ids))
    #
    # seq_len = np.round((np.mean(word_id_len) + 2 * np.std(word_id_len))).astype(int)
    #
    # word_id_train = sequence.pad_sequences(np.array(word_id_train), maxlen=seq_len, padding='pre')
    y_train_enc = np_utils.to_categorical(sentiment_train, num_labels)

    doc_terms_train, mid_terms, y_train, y_mid \
        = train_test_split(word_id_train, y_train_enc, test_size=SPLIT_SIZE, shuffle=True)

    doc_terms_val, doc_terms_test, y_val, y_test \
        = train_test_split(mid_terms, y_mid, test_size=0.5, shuffle=True)

    # LSTM
    print
    "fitting LSTM ..."
    model = Sequential()
    model.add(Embedding(VECTOR_SIZE, 128))
    model.add(SimpleRNN(128))
    model.add(Dense(num_labels))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(np.array(doc_terms_train), y_train, nb_epoch=3, batch_size=256, verbose=1)
    # validation_data=(np.array(doc_terms_val), y_val))

    # serialize model to JSON
    # model_json = model.to_json()
    # with open("./Model/LSTM_amazon_model_epoch1.json", "w") as json_file:
    #     json_file.write(model_json)
    # # serialize weights to HDF5
    # model.save_weights("./Model/LSTM_amazon_model_epoch1.h5")
    # print("Saved model to disk")

    # later...

    # # load json and create model
    # json_file = open('./Model/LSTM_amazon_model_epoch1.json', 'r')
    # loaded_model_json = json_file.read()
    # json_file.close()
    # model = model_from_json(loaded_model_json)
    # # load weights into new model
    # model.load_weights("./Model/LSTM_amazon_model_epoch1.h5")
    # print("Loaded model from disk")
    #
    # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # # Calculate the accuracy of testing data.
    # print("Start predicting.")
    test_pred = model.predict_classes(np.array(doc_terms_test), batch_size=256)
    print test_pred
    test_pred = np_utils.to_categorical(test_pred, num_labels)

    print "Accracy score:" + str(accuracy_score(test_pred, y_test))
    print "Confusion Matrix: "
    print confusion_matrix(test_pred.argmax(axis=1), y_test.argmax(axis=1))
