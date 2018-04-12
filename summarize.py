from sumy.parsers.html import HtmlParser
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer as Summarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

import reader
import csv

LANGUAGE = "english"
SENTENCES_COUNT = 10

reviewer = "Dennis+Schwartz"

outputFile = "./scaledata/"+reviewer+".tsv"

def firstSentence(content):
    sentences = content.split('.')
    return sentences[0]

def summarizeContent(content, sentences_count = 3):
    '''content: content to be summarized, sentence_count: num of sentence in summary '''
    parser = PlaintextParser.from_string(content, Tokenizer(LANGUAGE))
    # parser = PlaintextParser.from_file("document.txt", Tokenizer(LANGUAGE))
    stemmer = Stemmer(LANGUAGE)

    summarizer = Summarizer(stemmer)
    summarizer.stop_words = get_stop_words(LANGUAGE)
    
    summary = ""

    for sentence in summarizer(parser.document, sentences_count):
        summary += str(sentence)
    return summary

if __name__ == "__main__":
    # url = "http://www.zsstritezuct.estranky.cz/clanky/predmety/cteni/jak-naucit-dite-spravne-cist.html"
    # parser = HtmlParser.from_url(url, Tokenizer(LANGUAGE))
    # or for plain text files

    data = reader.readData(scale = 3)
    reviews = [entry[0] for entry in data[reviewer]]
    scales = [entry[1] for entry in data[reviewer]]
    print("data loaded")
    review = reviews[1]
    summaries = [summarizeContent(review, sentences_count=3) for review in reviews]
    print("sdfsd")

    with open(outputFile, 'w') as f:
        f.write('Review\tScore\n')
        for i in range(len(summaries)):
            f.write('{}\t{}\n'.format(summaries[i], scales[i]))


