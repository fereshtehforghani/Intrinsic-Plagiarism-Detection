from features import lexical, structural, syntax
from utils.utils import *


def init_dict():
    feature_dict = dict()
    for f in lexical.feature_name:
        feature_dict[f] = dict()
        for seg in lexical.seg_feature.keys():
            feature_dict[f][seg] = dict()
    for f in syntax.feature_name:
        feature_dict[f] = dict()
        for seg in syntax.seg_feature.keys():
            feature_dict[f][seg] = dict()
    for f in structural.feature_name:
        feature_dict[f] = dict()
        for seg in structural.seg_feature.keys():
            feature_dict[f][seg] = dict()
    return feature_dict


def extract_all_features(doc, doc_name):
    segmentations = ['g05-00', 'g09-00', 'g11-05', 'g20-10', 'g30-10', 's']
    feature_dict = init_dict()
    for seg in segmentations:
        segments = segment(doc, seg)
        lexical_features = lexical.extract_features(seg, segments)
        seq_1, seq_2 = syntax.find_all_sequence(doc)
        syntax_features = syntax.extract_features(doc, segment(doc, 's'), segments, seq_1, seq_2, seg)
        structural_features = structural.extract_features(seg, segments)
        for i, f_indx in enumerate(lexical.seg_feature[seg]):
            feature_dict[lexical.feature_name[f_indx]][seg][doc_name] = {'Feature': lexical_features[:, i]}
        for i, f_indx in enumerate(syntax.seg_feature[seg]):
            feature_dict[syntax.feature_name[f_indx]][seg][doc_name] = {'Feature': syntax_features[f_indx]}
        for i, f_indx in enumerate(structural.seg_feature[seg]):
            feature_dict[structural.feature_name[f_indx]][seg][doc_name] = {'Feature': structural_features[:, i]}
    return feature_dict


if __name__ == '__main__':
    doc = preprocess(read_file('../docs/1.txt'))
    print(extract_all_features(doc, '1'))
    # get_predict_data('g09-00', extract_all_features(doc, '1'), '1')
