from utils import preprocess

import matplotlib.pyplot as plt
from wordcloud import WordCloud

sub_a_label_mapping = {'NOT': 'Not Offensive', 'OFF': 'Offensive'}
sub_b_label_mapping = {'TIN': 'Targeted Insult', 'UNT': 'Untargeted'}
sub_c_label_mapping = {'IND': 'Individual', 'GRP': 'Group', 'OTH': 'Other'}


def plot_data_sample_bar_chart(sub_a_labels, sub_b_labels, sub_c_labels):
    fig, ax = plt.subplots()
    width = 0.35
    i_center = 0
    centers = [0 - width/2, 0 + width/2, 1 - width/2, 1 + width/2, 2 - width, 2, 2 + width]

    for k, v in sub_a_labels.items():
        ax.bar(centers[i_center], v, width, label=sub_a_label_mapping[k])
        i_center += 1
    for k, v in sub_b_labels.items():
        ax.bar(centers[i_center], v, width, label=sub_b_label_mapping[k])
        i_center += 1
    for k, v in sub_c_labels.items():
        ax.bar(centers[i_center], v, width, label=sub_c_label_mapping[k])
        i_center += 1

    ax.set_ylabel('Number')
    ax.set_title('Number of samples')
    ax.set_xticks(range(3), ['Offensive or not', 'Offense types', 'Offense target'])
    ax.legend()

    fig.tight_layout()

    plt.show()


def plot_tweet_char_length_hist(tweet_list):
    len_tweets = [len(tweet) for tweet in tweet_list]
    plt.hist(len_tweets, bins=30)
    plt.title('Number of characters')
    plt.show()
    

def plot_tweet_token_length_hist(tweet_list):
    len_tweets = [len(tweet) for tweet in tweet_list]
    plt.hist(len_tweets, bins=30)
    plt.title('Number of tokens')
    plt.show()

    
def plot_wordcloud(title, subtitles, *data_list):
    wc = WordCloud(width=1000, height=500, background_color="white", max_words=200, contour_width=0, random_state=5246)
    fig, ax = plt.subplots(len(data_list))
    if title: fig.suptitle(title)
    for i in range(len(data_list)):
        tweet_list = data_list[1]
        tweet_matrix, tweet_vocab = preprocess.count_vectorize(preprocess.join_as_sentence(tweet_list))
        word_freqs = dict(zip(tweet_vocab, tweet_matrix.A.sum(axis=0)))

        wc.generate_from_frequencies(word_freqs)

        ax[i].imshow(wc, interpolation="bilinear")
        ax[i].set_title(subtitles[i])
        ax[i].axis("off")
    
    plt.tight_layout()
    plt.show()