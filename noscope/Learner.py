import tempfile

import noscope
import numpy as np
from math import log, ceil


class Learner():
    def __init__(self, model, data, fname,
                 batch_size=256,
                 regression=False,
                 nb_examples=10000):
        self.model = model
        # Should take the form: X_train, Y_train, X_test, Y_test
        self.data = data
        self.batch_size = batch_size
        self.regression = regression
        self.temp_fname = tempfile.mkstemp(suffix='.hdf5', dir='/tmp/')[1]
        self.nb_examples = nb_examples
        self.fname = fname

    def get_inds(self):
        X_all, Y_all, _, _ = self.data
        return np.random.choice(
                range(len(X_all)), size=self.nb_examples, replace=False)

    def test_eval(self):
        _, _, X_test, Y_test = self.data
        if self.regression:
            return noscope.Models.evaluate_model_regression(self.model, X_test, Y_test)
        else:
            return noscope.Models.evaluate_model_multiclass(self.model, X_test, Y_test)

    def xvalid_eval(self):
        X_all, Y_all, _, _ = self.data
        test_inds = self.get_inds()
        X_test = X_all[test_inds]
        Y_test = Y_all[test_inds]
        metrics = self.model.evaluate(
            X_test, Y_test, batch_size=self.batch_size, verbose=0)
        # print metrics
        # print self.model.metrics_names
        return metrics[0]

    def run_iters(self, num_iter):
        X_all, Y_all, _, _ = self.data

        for i in xrange(num_iter):
            # TODO: for binary, balance classes
            train_inds = self.get_inds()

            X_train = X_all[train_inds]
            Y_train = Y_all[train_inds]
            self.model.fit(X_train, Y_train,
                           batch_size=32,
                           shuffle=False, # Already shuffled
                           class_weight='auto', # If already 50/50 doesn't do anything
                           nb_epoch=1) # TODO: figure out how to set the epoch #

        return self.xvalid_eval()

    def save_model(self, fname=None):
        if fname is None:
            self.model.save(self.fname)
        else:
            self.model.save(fname)

    def finish(self):
        pass
        # del self.model


class HyperBand():
    def __init__(self, max_iter):
        self.max_iter = max_iter

    def run(self, random_learner, top_n=5):
        eta = 3
        logeta = lambda x: log(x)/log(eta)
        s_max = int(logeta(self.max_iter))
        B = (s_max+1) * self.max_iter

        all_learners = []
        for s in reversed(range(s_max+1)):
            n = int(ceil(B / self.max_iter / (s+1) * eta**s))
            r = self.max_iter * eta**(-s)

            print 'Running outer loop loop with: %d %d' % (n, r)
            learners = [random_learner() for i in range(n)]
            for i in range(s+1):
                n_i = n * eta**(-i)
                r_i = int(r * eta**(i))
                print 'Running inner loop with: n_i=%d r_i=%d len(learners)=%d' % \
                    (n_i, r_i, len(learners))
                val_losses = [learner.run_iters(r_i) for learner in learners]

                # Unfortunate -_-
                to_take = int(n_i / eta)
                to_take += to_take == 0

                learners = [learners[i] for i in np.argsort(val_losses)[0:to_take]]
            all_learners.append(learners[0])

        val_losses = [learner.xvalid_eval() for learner in all_learners]
        learners = [all_learners[i] for i in np.argsort(val_losses)]

        return all_learners[0:top_n]
