import numpy as np
import pandas as pd

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import reader
import summarize

if __name__ == "__main__":
    SPLIT_SIZE = 0.3
    num_labels = 4
    reviewer = "James+Berardinelli"

    data = reader.readData(scale=num_labels)
    reviews = [entry[0] for entry in data[reviewer]]
    summaries = [summarize.summarizeContent(review, sentences_count=1) for review in reviews]
    
    raw_docs_train = summaries
    sentiment_train = [entry[1] for entry in data[reviewer]]

    # # load data
    # train_df = pd.read_csv('./kaggledata/train.tsv', sep='\t', header=0)

    # print train_df

    # raw_docs_train = train_df['Phrase'].values
    # sentiment_train = train_df['Sentiment'].values
    # num_labels = len(np.unique(sentiment_train))

    # print pd.value_counts(sentiment_train)
    # print num_labels

    # text pre-processing
    stop_words = set(stopwords.words('english'))
    stop_words.update(['.', ',', '"', "'", ':', ';', '(', ')', '[', ']', '{', '}'])
    stemmer = SnowballStemmer("english")

    processed_docs_train = []
    for doc in raw_docs_train:
        tokens = word_tokenize(doc)
        filtered = [word for word in tokens if word not in stop_words]
        stemmed = [stemmer.stem(word) for word in filtered]
        processed_docs_train.append(stemmed)
    
    with open('test.txt', 'w') as f:
        for line in processed_docs_train:
            f.write(str(line) + '\n')