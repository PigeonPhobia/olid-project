from collections import Counter, OrderedDict

from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import torch
import torch.nn.functional as F

from torchtext.vocab import vocab

from torch.nn.utils.rnn import pad_sequence

import spacy

nlp = spacy.load("en_core_web_sm")


def nltk_tokenize(tweet_list):
    tweet_tokenizer = TweetTokenizer()
    return [tweet_tokenizer.tokenize(tweet) for tweet in tweet_list]


def spacy_pipeline(tweet_list):
    return [nlp(tweet) for tweet in tweet_list]


def spacy_normalize(doc_list, stop_removal=True, lemmatized=True, punct_removal=True):
    res = []
    key = "lemma_" if lemmatized else "text"
    to_remove_stop = lambda t: t.is_stop if stop_removal else False
    to_remove_punct = lambda t: t.pos_ in ['PUNCT'] if punct_removal else False
    for doc in doc_list:
        res.append([getattr(t, key).lower() for t in doc if (not to_remove_stop(t) and not to_remove_punct(t))])
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


def create_fixed_length_batch(sequences, length):
    
    # Pad sequences w.r.t. longest sequences
    sequences_padded = pad_sequence(sequences,  batch_first=True, padding_value=0)

    # Get the current sequence length
    max_seq_len = sequences_padded.shape[1]
    
    if max_seq_len > length:
        # Cut sequences if to0 long
        return sequences_padded[:,:length]
    else:
        # Pad sequences if too short
        return F.pad(sequences_padded, (0, length-max_seq_len), mode="constant", value=0)


def transform_word_to_vector(documents, num_vocab=10000, num_tokens=30):
    token_counter = Counter()
    for text in documents:
        for token in text:
            token_counter[token] += 1

    token_counter_sorted = sorted(token_counter.items(), key=lambda x: x[1], reverse=True)
    token_ordered_dict = OrderedDict(token_counter_sorted[:num_vocab])
    
    SPECIALS = ["<PAD>", "<UNK>", "<SOS>", "<EOS>"]
    vocabulary = vocab(token_ordered_dict, specials=SPECIALS)
    vocabulary.set_default_index(vocabulary["<UNK>"])
    
    documents_vector = [torch.tensor(vocabulary.lookup_indices(text)) for text in documents]
    
    documents_padded = create_fixed_length_batch(documents_vector, num_tokens)
    
    return vocabulary, documents_padded