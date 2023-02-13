from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import spacy

nlp = spacy.load("en_core_web_sm")


def nltk_tokenize(tweet_list):
    tweet_tokenizer = TweetTokenizer()
    return [tweet_tokenizer.tokenize(tweet) for tweet in tweet_list]


def spacy_pipeline(tweet_list):
    return [nlp(tweet) for tweet in tweet_list]


def spacy_normalize(doc_list, stop_removal=True, lemmatized=True):
    res = []
    key = "lemma_" if lemmatized else "text"
    to_remove = lambda t: t.is_stop if stop_removal else False
    for doc in doc_list:
        res.append([getattr(t, key).lower() for t in doc if (not to_remove(t) and t.pos_ not in ['PUNCT'])])
    return res


def join_as_sentence(tweet_list):
    return [" ".join(tweet) for tweet in tweet_list]


def count_vectorize(documents):
    count_vectorizer = CountVectorizer()
    X_tf = count_vectorizer.fit_transform(documents)
    return X_tf, count_vectorizer.get_feature_names_out()


def remove_user_mask(documents):
    res = []
    for doc in documents:
        res.append([t for t in doc if t != "@user"])
    return res