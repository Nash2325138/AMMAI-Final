import numpy as np
import torch
import scipy.spatial.distance.cosine as cosine_dist


# copied from https://github.com/TreB1eN/InsightFace_Pytorch/blob/master/verifacation.py
def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size
    return tpr, fpr, acc


class Evaluator():
    source_emb = []
    target_emb = []
    is_same = []
    prepared = False

    def extend(self, source_emb, target_emb, is_same):
        """
        Collect prediction.
        """
        for container, element in zip(
            [self.source_emb, self.target_emb, self.is_same], [source_emb, target_emb, is_same]
        ):
            if torch.is_tensor(element):
                element = element.cpu().numpy()
            elif isinstance(element, np.ndarray):
                0  # do nothing
            elif isinstance(element, list):
                element = np.array(element)
            else:
                raise NotImplementedError()
            container.append(element)
        self.prepared = False

    def prepare(self):
        """
        This function concatenates the collected datas.
        """
        if self.prepared:
            return
        self.source_embeddings = np.concatenate(self.source_emb, axis=0)
        self.target_embedggins = np.concatenate(self.target_emb, axis=0)
        self.is_sames = np.concatenate(self.is_same, axis=0)
        self.prepared = True

    def _calculate_dist(self, strategy='l2_dist'):
        assert strategy in ['l2_dist', 'cosine']
        if strategy == 'l2_dist':
            diff = np.subtract(self.source_embeddings, self.target_embedggins)
            dist = np.sum(np.square(diff), 1)
        else:
            dist = [cosine_dist(self.source_embeddings[i], self.target_embedggins[i])
                    for i in range(len(self.source_embeddings))]
            dist = np.stack(dist, axis=0)
        return dist

    def _calculate_thresholds(self, strategy='l2_dist', n_thres=400):
        assert strategy in ['l2_dist', 'cosine']
        if strategy == 'l2_dist':
            thresholds = np.arange(0, 4, 4 / n_thres)
        else:
            thresholds = np.arange(0, 2, 2 / n_thres)
        assert(len(thresholds) == n_thres)
        return thresholds

    def calculate_roc(self, strategy='l2_dist', n_thres=400):
        """
        Calculate points on the ROC curve, return following numbers
        (true positive rates, false positive rates, all accuraries, best threshold)
        The best threshold is determined by the highest accuracy.
        """
        assert strategy in ['l2_dist', 'cosine']
        self.prepare()
        thresholds = self._calculate_thresholds(strategy, n_thres)
        dists = self._calculate_dist(strategy)

        # some codes are modified from https://github.com/TreB1eN/InsightFace_Pytorch/blob/master/verifacation.py
        tprs = np.zeros(n_thres)
        fprs = np.zeros(n_thres)
        accs = np.zeros(n_thres)

        for i, threshold in enumerate(thresholds):
            tprs[i], fprs[i], accs[i] = calculate_accuracy(threshold, dists, self.is_sames)

        best_threshold_index = np.argmax(accs)
        best_threshold = thresholds[best_threshold_index]
        return tprs, fprs, accs, best_threshold
