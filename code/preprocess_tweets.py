import nltk
import string
import requests
import numpy as np
import pandas as pd
import re, sys, csv, json
from textblob import TextBlob
from collections import defaultdict
from nltk.tokenize import TweetTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import PorterStemmer, WordNetLemmatizer

# local repos
import signals
import words_lists
# importing word breaker
from word_breaking import word_breaker


printable = set(string.printable)

punctuation = list(string.punctuation)
punctuation.remove("-")
punctuation.remove('_')

stoplist = words_lists.long_stop_list + punctuation + words_lists.slang_abbreviations

nltk_tok = TweetTokenizer(preserve_case=True, reduce_len=True, strip_handles=True)


'''def get_time_in_tweet(text):
    # CURL
    '''
    #wget --post-data 'i woke up 3:00 pm.' 'localhost:9000/?properties={"tokenize.whitespace":"true","annotators":"tokenize,ssplit,pos,lemma,ner","outputFormat":"json"}' -O -
'''

    url = 'http://localhost:9000/?properties={"annotators": "tokenize,tokenize,ssplit,pos,lemma,ner", "outputFormat": "json"}'

    r = requests.post(url, data=text)

    r.raise_for_status()

    times = list()

    try:
        for sent in r.json()["sentences"]:
            for token in sent["tokens"]:
                if token["ner"] == "TIME":
                    times.append(token["word"])

    except:
        pass

    return times'''

def preprocess_tweet(tweet):

    #tweet = tweet.replace("'s","")

    # this will replace seeds (as phrases) as unigrams. lack of > lack_of
    for seed in signals.all_seeds:
        if seed in tweet and " " in seed:
            tweet = tweet.replace(seed, seed.replace(" ", "_"))

    # remove retweet handler
    if tweet[:2] == "RT":
        colon_idx = tweet.index(":")
        tweet = tweet[colon_idx+2:]

    # remove url from tweet
    tweet = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', tweet)

    # remove non-ascii characters
    tweet = filter(lambda x: x in printable, tweet)

    # additional preprocessing
    tweet = tweet.replace("\n", " ").replace(" https","").replace("http","")

    # remove all mentions in tweet
    mentions = re.findall(r"@\w+", tweet)
    for mention in mentions:
        tweet = tweet.replace(mention, "")

    # break usernames and hashtags +++++++++++++
    for term in re.findall(r"#\w+", tweet):

        token = term[1:]

        # remove any punctuations from the hashtag and mention
        # ex: Troll_Cinema => TrollCinema
        token = token.translate(None, ''.join(string.punctuation))

        segments = word_breaker.segment(token)
        segments = ' '.join(segments)

        tweet = tweet.replace(term, segments)

    # remove all punctuations from the tweet text
    tweet = "".join([char for char in tweet if char not in punctuation])

    # remove trailing spaces
    tweet = tweet.strip()

    # replace time mention with time
    '''for time in get_time_in_tweet(tweet):
        tweet = tweet.replace(time, "time")'''

    # remove all tokens in the tweet where the token is
    # a stop word or an emoji
    tweet = [word.lower() for word in nltk_tok.tokenize(tweet) if word.lower() not in stoplist and word.lower() not in words_lists.emojies and len(word) > 1]

    tweet = " ".join(tweet)

    # remove numbers
    tweet = re.sub(r'[\d-]+', 'NUM', tweet)
    # padding NUM with spaces
    tweet = tweet.replace("NUM", " NUM ")
    # remove multiple spaces in tweet text
    tweet = re.sub('\s{2,}', ' ', tweet)

    print tweet

    return tweet


def preprocess(account_file):

    # CSV file:
    #   0:Tweet_ID     1:Raw_text    2:Cleaned_text
    #   3:Created_at   4:Sentiment   5:Annotation

    fileName = "Data/Raw/"+account_file

    account_tweets = pd.read_csv(fileName)
    account_tweets = account_tweets.replace(np.nan,' ', regex=True)

    for index, tweet in account_tweets.iterrows():

        sent_score = TextBlob(tweet.Raw_text.decode('ascii', errors="ignore")).sentiment.polarity

        cleaned_text = preprocess_tweet(tweet.Raw_text)

        account_tweets.set_value(index, "Sentiment", str(sent_score))
        account_tweets.set_value(index, "Cleaned_text", str(cleaned_text))


    account_tweets.to_csv("Data/Preprocessed/"+account_file+"_cleaned.csv", sep=',')

if __name__ == "__main__":

    preprocess("SuicidalIdeas_ankita.csv")
