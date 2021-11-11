import gzip
import pickle
import numpy as np
from features.extract_features import extract_all_features
from utils.utils import read_file, preprocess
import outlier_detection.model as od


def get_predict_data(seg, features, doc_name):
    X = []
    for f in features:
        if features[f][seg] != {}:
            f_list = []
            f_list += list(features[f][seg][doc_name]["Feature"])
            X.append(f_list)
    return np.asarray(list(map(list, zip(*X))))


def classifier_score(classifier, segment, test_data):
    file = gzip.open('train_models/{}/{}.gz'.format(classifier, segment), 'rb')
    model = pickle.load(file)
    return model.predict(test_data)


def predict_by_regressor(segment, test_data):
    file = gzip.open('train_models/mlp_regressor/' + segment + ".gz", "rb")
    model = pickle.load(file)
    predict = model.predict(test_data)
    return predict


def predict_by_classifier(segment, test_data):
    file = gzip.open('train_models/mlp_classifier/' + segment + ".gz", "rb")
    model = pickle.load(file)
    predict = model.predict(test_data)
    return predict


def predict_by_GBRT(segment, test_data):
    file = gzip.open('train_models/gbrt/' + segment + ".gz", "rb")
    model = pickle.load(file)
    predict = model.predict(test_data)
    return predict


def predict_by_RF(segment, test_data):
    file = gzip.open('train_models/rf/' + segment + ".gz", "rb")
    model = pickle.load(file)
    predict = model.predict(test_data)
    return predict


def predict(file_path, file_name, segmentations, classifier):
    doc = preprocess(read_file(file_path))
    features = extract_all_features(doc, file_name)
    score = {}
    for seg in segmentations:
        test_data = get_predict_data(seg, features, file_name)
        clf_score = classifier_score(classifier, seg, test_data)
        score = od.predict(doc, score, seg, clf_score)
    y_token_pred = od.get_y_token(score)
    y_token_pred_bin = list(map(int, np.round(y_token_pred)))
    return y_token_pred_bin, doc


if __name__ == '__main__':
    doc_path = input("Please enter doc path:\nSample input: docs/1.txt\n")
    seg = ['g05-00', 'g09-00', 'g11-05', 'g20-10', 'g30-10', 's']
    y, doc = predict(doc_path, '1', seg, 'gbrt')
    # save new doc
    with open("docs/test.txt", "a") as myfile:
        myfile.write(str(doc))
    suspicious = []
    for i, l in enumerate(y):
        if l:
            suspicious += [i]
    # cluster words
    cluster = []
    temp = []
    for i in range(len(suspicious) - 1):
        if suspicious[i + 1] - suspicious[i] < 5:
            temp += [suspicious[i]]
        else:
            temp += [suspicious[i]]
            cluster += [temp]
            temp = []
    if temp:
        if suspicious[-1] - suspicious[-2] < 5:
            cluster += [temp + [suspicious[-1]]]
        else:
            cluster += [temp]
            cluster += [suspicious[-1]]
    # make suspicious
    with open("docs/suspicious_test.txt", "a") as file:
        file.write(str(cluster))
    print("Complete prediction")
    print("New doc and suspicious token numbers are saved in docs folder :)")
