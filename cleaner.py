import re
import gensim
from singleton import Singleton

class Cleaner(metaclass=Singleton):

    def remove_stopwords_punctuations(self, phrase):
        phrase_split = gensim.utils.simple_preprocess(phrase, deacc=True)
        sentence = " ".join(phrase_split).strip()
        return sentence


    def remove_url(self, phrase):
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        phrase = url_pattern.sub(r'', phrase)
        return phrase


    def remove_emails(self, phrase):
        return re.sub('\S*@\S*\s?', '', phrase)


    def remove_one_or_more_space_char(self, phrase):
        return re.sub('\s+', ' ', phrase)


    def remove_single_quotes(self, phrase):
        return re.sub("\'", "", phrase)

