import matplotlib.pyplot as plt

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