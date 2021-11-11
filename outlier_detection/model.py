import numpy as np
from sklearn import metrics
from sklearn.neighbors import BallTree
from utils.utils import segment


class KNN:
    def __init__(self, n_neighbors=5, threshold=0.5):
        self.n_neighbors = n_neighbors
        self.threshold = threshold
        self.tree_ = None
        self.distances_ = None
        self.labels_ = None

    def fit(self, X):
        self.tree_ = BallTree(X)
        self.distances_ = self.tree_.query(X, k=self.n_neighbors + 1)[0][:, -1]
        self.labels_ = np.where(self.distances_ > self.threshold, 1, 0)
        return self.labels_

    def get_params(self):
        return {'n_neighbors': self.n_neighbors,
                'threshold': round(self.threshold, 4)}


class ThresholdBased:
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.labels_ = None

    def fit(self, X):
        self.labels_ = np.where(X > self.threshold, 1, 0)
        return self.labels_

    def get_params(self):
        return {'threshold': round(self.threshold, 4)}


def eval_model(seg_name, model, f1_score):
    print('---------------------------------------------------------')
    print(f'Segmentation Name: {seg_name}')
    print(f'Model Name: {model.__class__.__name__}')
    print(f'Params:', str(model.get_params()).replace(', ', '\n\t '))
    print(f'Mean F1-Score Train: {np.mean(f1_score):.6f}')


best_model = {'g11-05': ThresholdBased(0.7),
              'g20-10': ThresholdBased(0.8),
              'g30-10': ThresholdBased(0.8),
              'g09-00': ThresholdBased(0.7),
              'g05-00': ThresholdBased(0.7),
              's': ThresholdBased(0.7)}


def combine_scores(doc, score, segment_name, y_pred):
    segments = segment(doc, segment_name)
    for i, seg in enumerate(segments):
        for token in seg:
            mean, cnt = score.get(token.idx, (0, 0))
            score[token.idx] = [(mean * cnt + y_pred[i]) / (cnt + 1), cnt + 1]
    return score


def get_y_token(score):
    y_token_pred = np.vstack((list(score.values())))[:, 0].tolist()
    return y_token_pred


def predict(doc, score, segment_name, clf_score):
    model = best_model[segment_name]
    score = combine_scores(doc, score, segment_name, model.fit(clf_score))
    return score

