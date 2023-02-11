from nltk.tokenize import TweetTokenizer

def nltk_tokenize(tweet_list):
    tweet_tokenizer = TweetTokenizer()
    return [tweet_tokenizer.tokenize(tweet) for tweet in tweet_list]