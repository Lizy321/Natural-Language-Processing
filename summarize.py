from sumy.parsers.html import HtmlParser
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer as Summarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

LANGUAGE = "english"
SENTENCES_COUNT = 10

inputFile = "./scaledata/Dennis+Schwartz/subj.Dennis+Schwartz"
outputFile = "./scaledata/Dennis+Schwartz/sum.Dennis+Schwartz"

if __name__ == "__main__":
    # url = "http://www.zsstritezuct.estranky.cz/clanky/predmety/cteni/jak-naucit-dite-spravne-cist.html"
    # parser = HtmlParser.from_url(url, Tokenizer(LANGUAGE))
    # or for plain text files



    review = "in my opinion , a movie reviewer's most important task is to offer an explanation for his opinion . but with soul survivors , i'm so full of critical rage that this review is liable to turn into a venomous , uncontrollable rant , obligations damned . however , protocol forces me to do otherwise . rarely have i seen a director so thoroughly mishandle material . in his directorial debut , steve carpenter does nothing to provoke the audience into feeling any emotion . when the movie's camerawork isn't pedestrian , it's clich ? d . carpenter's need for close-ups , his camera following sagemiller from behind , drain the movie of any anticipation . we already know what to look for , so why should we be surprised ? and the sense of scary atmosphere , which was recently done so well in the others , is nowhere to be found here . carpenter's script relies on random twists and turns with a minimum of logic and loads of laziness . this is a movie where the ending features one character providing an explanation about the plot to another character , which is one of the lamest screenwriting devices around . and guess what ? this movie has both ! not that the main body of the film , which has sagemiller constantly running in fear or having a nervous breakdown , is anything worthwhile . since there's no captivating dialogue , no character chemistry exists anywhere . that's a huge problem , since the four main characters are supposed to be couples . carpenter can't even get the smutty scenes right , which are becoming more prevalent in pg-13 movies like bring it on and get over it . [the film was recently re-cut from an r rating to get more kids in the seats . -ed . ] when sagemiller and dushku dance together at a club , he rarely shows them in a full shot and he never keeps the camera on them for longer than a second before relating to some michael bay-style quick cuts . regardless , any sexiness in that scene is undercut by its stupidity . why wouldn't they wash the clothes in the sink or in the washing machine ? the cast , which will see better material in the future , would be wise to leave this one off their resumes . i felt sorry for affleck , who i've liked in other movies , and bentley , who was great in american beauty . and i'm not even getting into luke wilson's role as a priest . bottom line : soul survivors is so awful i feel compelled to knock on doors and warn people about it . rating : * [lowest rating] |------------------------------| \ * * * * * perfection \ \ * * * * good , memorable film \ \ * * * average , hits and misses \ \ * * sub-par on many levels \ \ * unquestionably awful \ |------------------------------| mpaa rating : pg-13 "
    parser = PlaintextParser.from_string(review, Tokenizer(LANGUAGE))
    # parser = PlaintextParser.from_file("document.txt", Tokenizer(LANGUAGE))
    stemmer = Stemmer(LANGUAGE)

    summarizer = Summarizer(stemmer)
    summarizer.stop_words = get_stop_words(LANGUAGE)
    
    for sentence in summarizer(parser.document, SENTENCES_COUNT):
        print(sentence)
        

