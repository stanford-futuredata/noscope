import bisect
import csv
import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Assumes two classes
def plot_auc_pr(Y_true, Y_probs,
                roc_fname=None, pr_fname=None,
                fnr_fpr_fname=None):
    average_precision = sklearn.metrics.average_precision_score(Y_true, Y_probs)
    if pr_fname is not None:
        precision, recall, thresholds = sklearn.metrics.precision_recall_curve(Y_true, Y_probs)
        plt.clf()
        plt.plot(recall, precision, label='Precision-Recall curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('Precision-Recall example: AUC = %0.2f' % (average_precision))
        plt.savefig(pr_fname)

    fpr, tpr, thresholds = sklearn.metrics.roc_curve(Y_true, Y_probs)
    roc_auc = sklearn.metrics.auc(fpr, tpr)
    if roc_fname is not None:
        plt.clf()
        plt.plot(fpr, tpr,
                 label='ROC (area = %0.2f)' % (roc_auc))
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.savefig(roc_fname)

    fnr = 1 - tpr
    if fnr_fpr_fname is not None:
        plt.clf()
        plt.plot(fnr, fpr, label='FNR vs FPR')
        plt.xlim([-0.01, .10])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Negative Rate')
        plt.ylabel('False Positive Rate')
        plt.title('FNR vs FPR')
        plt.savefig(fnr_fpr_fname)

    return roc_auc, average_precision


def windowed_accuracy(Y_pred, Y_true, WINDOW=15, THRESH=10):
    def to_single(Y):
        if len(Y.shape) > 1:
            assert len(Y.shape) == 2
            assert Y.shape[1] <= 2
            if Y.shape[1] == 2:
                return Y[:, 1]
            if Y.shape[1] == 1:
                return np.ravel(Y)
        return Y

    def process(Y):
        Y = to_single(Y)
        to_take = WINDOW * (len(Y) / WINDOW)
        Y = np.reshape(Y[:to_take], (len(Y) / WINDOW, WINDOW))
        Y = np.add.reduce(Y, axis=1) > THRESH
        Y = Y.reshape((len(Y), 1)) # for scikit learn
        return Y

    Y_pred = process(Y_pred)
    Y_true = process(Y_true)
    precision, recall, fbeta, support = sklearn.metrics.precision_recall_fscore_support(
        Y_pred, Y_true)
    accuracy = sklearn.metrics.accuracy_score(Y_pred, Y_true)
    metrics = {'precision': precision,
               'recall': recall,
               'fbeta': fbeta,
               'support': support,
               'accuracy': accuracy}
    # FIXME
    return accuracy, support

# Assumes two classes
# TODO: check that this is actually correct
def yolo_oracle(Y_true, Y_probs, fpr_thresh=0.01, fnr_thresh=0.01):
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(Y_true, Y_probs)
    fnr = 1 - tpr

    lower_ind = 0
    upper_ind = len(Y_true)

    for i in xrange(len(fpr)):
        if fpr[i] > fpr_thresh:
            upper_ind = bisect.bisect_right(Y_probs, thresholds[i])
            upper_ind = min(upper_ind, len(Y_probs) - 1)
            upper_ind = max(upper_ind, 0)
            break
    for i in xrange(len(fnr) - 1, -1 , -1):
        if fnr[i] > fnr_thresh:
            lower_ind = bisect.bisect_left(Y_probs, thresholds[i])
            lower_ind = min(lower_ind, len(Y_probs) - 1)
            lower_ind = max(lower_ind, 0)
            break

    return upper_ind - lower_ind, Y_probs[lower_ind], Y_probs[upper_ind]

def output_csv(csv_fname, stats, headers, quoting=csv.QUOTE_MINIMAL):
    df = pd.DataFrame(stats, columns=headers)
    df.to_csv(csv_fname, index=False, quoting=quoting)


class OutputRecorder:
    def __init__(self, csv_in):
        self.csv_in = csv_in
        self.headers = ['frame', 'labels']
        self.counter = 0
        self.rows = []

    def add_row(self, val, object_name):
        obj = {'object_name': object_name} if val else {}
        if len(obj) == 0: # empty dict
            self.rows.append([ self.counter,  '"[]"' ])
        else:
            self.rows.append([ self.counter,  '"[' + str(obj) + ']"' ])
        self.counter += 1

    def output_csv(self):
        output_csv(self.csv_in, self.rows, self.headers, quoting=csv.QUOTE_NONE)

