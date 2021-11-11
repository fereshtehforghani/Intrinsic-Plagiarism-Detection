import numpy as np
from utils.utils import normalize


def mean_word_length(seg):
    words = [token for token in seg if not token.is_punct]
    if not words:
        return 0
    length = sum([len(word) for word in words])
    return length / len(words)


def punctuation_count(seg):
    puncts = [token for token in seg if token.is_punct]
    return len(puncts) / len(seg)


def sentence_length(seg):
    words = [token for token in seg if not token.is_punct]
    return len(words)


def stopword_count(seg):
    stops = [token for token in seg if token.is_stop]
    return len(stops) / len(seg)


feature_name = ['Mean_Word_Length', 'Punctuation_Count', 'Sent_Length', 'Stopword_Count']

feature_func = [mean_word_length, punctuation_count, sentence_length, stopword_count]

seg_feature = {'g11-05': [0, 3],
               'g20-10': [0, 3],
               'g30-10': [0, 3],
               'g09-00': [0, 3],
               'g05-00': [3],
               's': [0, 1, 2, 3]}


def extract_features(segmentation, segments):
    f_index = seg_feature[segmentation]
    X = np.empty((len(segments), len(f_index)))
    for i, seg in enumerate(segments):
        X[i] = [feature_func[j](seg) for j in f_index]
    return normalize(X)
