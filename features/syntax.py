from __future__ import unicode_literals, print_function
from itertools import islice
import numpy as np
import spacy

from utils.utils import segment_patter, normalize_2

nlp = spacy.load('en_core_web_sm')
nlp.max_length = 5000000


def find_sentences(text):
    return [i for i in text.sents]


def tag_count(segments, tag_name):
    t_count = []
    for text in segments:
        tags = [token.tag_ for token in text]
        count = 0
        for i in tags:
            if i == tag_name:
                count += 1
        t_count.append(count)
    return normalize_2(np.array(t_count))


def count_function_Words(segments):
    functional_words = """a between in nor some upon
    about both including nothing somebody us
    above but inside of someone used
    after by into off something via
    all can is on such we
    although cos it once than what
    am do its one that whatever
    among down latter onto the when
    an each less opposite their where
    and either like or them whether
    another enough little our these which
    any every lots outside they while
    anybody everybody many over this who
    anyone everyone me own those whoever
    anything everything more past though whom
    are few most per through whose
    around following much plenty till will
    as for must plus to with
    at from my regarding toward within
    be have near same towards without
    because he need several under worth
    before her neither she unless would
    behind him no should unlike yes
    below i nobody since until you
    beside if none so up your
    """

    functional_words = functional_words.split()
    f_count = []
    for text in segments:
        count = 0
        if len(text) == 0:
            return 0
        words = [token for token in text]
        for i in words:
            if str(i) in functional_words:
                count += 1
        f_count.append(count)
    return normalize_2(f_count)


def find_subject_verb_object(parsed_text):
    subject_pos = -1
    object_pos = -1
    positions = [(token, token.i) for token in parsed_text]
    for t in positions:
        if t[0].dep_ == "nsubj":
            subject_pos = t[1] - positions[0][1]
        if t[0].dep_ == "dobj" or t[0].dep_ == "iobj":
            object_pos = t[1] - positions[0][1]
    return subject_pos, object_pos


def subject_pos(sents):
    subject_positions = []
    for sent in sents:
        subject_positions.append(find_subject_verb_object(sent)[0]/len(sent))
    return normalize_2(subject_positions)


def object_pos(sents):
    object_positions = []
    for sent in sents:
        object_positions.append(find_subject_verb_object(sent)[0] / len(sent))
    return normalize_2(object_positions)


def all_sequence(tags, k, seq):
    for i in range(len(tags)):
        if k == 1:
            seq.append(tags[i])
        elif k == 2:
          if i < len(tags) - 1:
            seq.append((tags[i], tags[i + 1]))


def frequency_most_common_seq(text, k_most, all_seq, n, n_gram, overlap):
    tags = [token.tag_ for token in text]
    count = dict()
    for i in all_seq:
        count[i] = count.get(i, 0) + 1
    most_common = list(
        islice({k: v for k, v in sorted(count.items(), key=lambda item: item[1], reverse=True)}.items(), 0, k_most))
    frequency = []
    start_tag, end_tag = 0, 0
    length = len(tags)
    while end_tag != length:
        end_tag = min(length, start_tag + n_gram)
        tag_list_1 = []
        tag_list_2 = []
        temp = [0 for _ in range(k_most)]
        if n == 1:
            all_sequence(tags[start_tag : end_tag], 1, tag_list_1)
            for i in range(k_most):
                temp[i] += tag_list_1.count(most_common[i][0])
        elif n == 2:
            all_sequence(tags[start_tag : end_tag], 2, tag_list_2)
            for i in range(k_most):
                temp[i] += tag_list_2.count(most_common[i][0])
        frequency.append(temp)
        start_tag += (n_gram - overlap)
    return frequency


# def find_all_sequence(plagiarised, non_plagiarised):
#     all_seq_len_1 = []
#     all_seq_len_2 = []
#     p_tags = [token.tag_ for token in plagiarised]
#     np_tags = [token.tag_ for token in non_plagiarised]
#     all_sequence(p_tags, 1, all_seq_len_1)
#     all_sequence(np_tags, 1, all_seq_len_1)
#     all_sequence(p_tags, 2, all_seq_len_2)
#     all_sequence(np_tags, 2, all_seq_len_2)
#     return all_seq_len_1, all_seq_len_2

def find_all_sequence(doc):
    all_seq_len_1 = []
    all_seq_len_2 = []
    tags = [token.tag_ for token in doc]
    all_sequence(tags, 1, all_seq_len_1)
    all_sequence(tags, 2, all_seq_len_2)
    return all_seq_len_1, all_seq_len_2


def extract_features(text, sentences, segments, all_seq_len_1, all_seq_len_2, segmentation):
    PRP = tag_count(segments, "PRP")
    function_words = count_function_Words(segments)
    subject_positions = subject_pos(sentences)
    object_positions = object_pos(sentences)
    # for sent in sentences:
    #     position = find_subject_verb_object(sent)
    #     subject_positions.append(position[0]/len(sent))
    #     object_positions.append(position[1]/len(sent))
    # for seg in segments:
    #     PRP.append(tag_count(seg, 'PRP'))
    #     function_words.append(count_function_Words(seg))
    # if segmentation in ["g05-00", 'g05-00', 'g09-00', 'g11-05']:
    #     return np.array([tag_count(segments, "PRP"), count_function_Words(segments)], dtype=object)
    # elif segmentation == "s":
    #     return np.array([subject_pos(sentences), object_pos(sentences)], dtype=object)
    # elif segmentation in ['g20-10', 'g30-10']:
    _, _, n_gram, overlap = segment_patter.search(segmentation).groups()
    if n_gram == "20" or n_gram == "30":
        len_1_seq = frequency_most_common_seq(text, 5, all_seq_len_1, 1, int(n_gram), int(overlap))
        len_2_seq = frequency_most_common_seq(text, 5, all_seq_len_2, 2, int(n_gram), int(overlap))
        len_1_seq_new = []
        len_2_seq_new = []
        for j in range(5):
            len_1 = list(x[j] for x in len_1_seq)
            len_2 = list(x[j] for x in len_2_seq)
            len_1_seq_new.append(normalize_2(len_1))
            len_2_seq_new.append(normalize_2(len_2))
    else:
        len_1_seq_new = [[], [], [], [], []]
        len_2_seq_new = [[], [], [], []]
    return np.array([PRP, function_words, subject_positions, object_positions, len_1_seq_new[0], len_1_seq_new[1],
                     len_1_seq_new[2], len_1_seq_new[3], len_1_seq_new[4], len_2_seq_new[0], len_2_seq_new[1],
                     len_2_seq_new[2], len_2_seq_new[3]], dtype=object)


feature_name = ['PRP_Count', 'Function_Word_Count', 'Object_Position', 'Subject_Position', 'MC_Sequence_1_1',
                'MC_Sequence_1_2', 'MC_Sequence_1_3', 'MC_Sequence_1_4', 'MC_Sequence_1_5',
                'MC_Sequence_2_1', 'MC_Sequence_2_2', 'MC_Sequence_2_3', 'MC_Sequence_2_4']


seg_feature = {'g11-05': [0, 1],
               'g20-10': [4, 5, 6, 7, 8, 9, 10, 11, 12],
               'g30-10': [4, 5, 6, 7, 8, 9, 10, 11, 12],
               'g09-00': [0, 1],
               'g05-00': [0, 1],
               's': [2, 3]}
