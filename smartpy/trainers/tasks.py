# -*- coding: utf-8 -*-
from __future__ import division

import os
import numpy as np
from collections import OrderedDict
from time import time


class StoppingCriterion(object):
    def check(self, status):
        raise NotImplementedError("Subclass has to implement this function.")


class Task(object):
    def __init__(self):
        self.updates = OrderedDict()

    def init(self, status):
        pass

    def pre_epoch(self, status):
        pass

    def pre_update(self, status):
        pass

    def post_update(self, status):
        pass

    def post_epoch(self, status):
        pass

    def finished(self, status):
        pass

    def execute(self, status):
        pass


class View(Task):
    def __init__(self):
        super(View, self).__init__()
        self.value = None
        self.last_update = -1

    def view(self, status):
        if self.last_update != status.current_update:
            self.update(status)
            self.last_update = status.current_update

        return self.value

    def update(self, status):
        raise NotImplementedError("Subclass has to implement this function.")

    def __str__(self):
        return "{0}".format(self.value)


class Print(Task):
    def __init__(self, view, msg="{0}", each_epoch=1, each_update=0):
        super(Print, self).__init__()
        self.msg = msg
        self.each_epoch = each_epoch
        self.each_update = each_update
        self.view_obj = view

        # Get updates of the view object.
        self.updates.update(view.updates)

    def post_update(self, status):
        self.view_obj.post_update(status)

        if self.each_update != 0 and status.current_update % self.each_update == 0:
            value = self.view_obj.view(status)
            print self.msg.format(value)

    def post_epoch(self, status):
        self.view_obj.post_epoch(status)

        if self.each_epoch != 0 and status.current_epoch % self.each_epoch == 0:
            value = self.view_obj.view(status)
            print self.msg.format(value)

    def init(self, status):
        self.view_obj.init(status)

    def pre_epoch(self, status):
        self.view_obj.pre_epoch(status)

    def pre_update(self, status):
        self.view_obj.pre_update(status)

    def finished(self, status):
        self.view_obj.finished(status)


class PrintEpochDuration(Task):
    def __init__(self):
        super(PrintEpochDuration, self).__init__()

    def init(self, status):
        self.training_start_time = time()

    def pre_epoch(self, status):
        self.epoch_start_time = time()

    def post_epoch(self, status):
        print "Epoch {0} done in {1:.03f} sec.".format(status.current_epoch, time() - self.epoch_start_time)

    def finished(self, status):
        print "Training done in {:.03f} sec.".format(time() - self.training_start_time)


class MaxEpochStopping(StoppingCriterion):
    def __init__(self, nb_epochs_max):
        self.nb_epochs_max = nb_epochs_max

    def check(self, status):
        return status.current_epoch > self.nb_epochs_max


class Evaluate(View):
    def __init__(self, func):
        super(Evaluate, self).__init__()
        self.func = func

    def update(self, status):
        self.value = self.func()


class AverageNLL(Evaluate):
    def __init__(self, nll, dataset, batch_size=None):
        import theano
        import theano.tensor as T

        if batch_size is None:
            batch_size = len(dataset)

        nb_batches = int(np.ceil(len(dataset[0]) / batch_size))
        dataset = theano.shared(dataset, name='data', borrow=True)

        input = T.matrix('input')
        no_batch = T.iscalar('no_batch')
        compute_nll = theano.function([no_batch],
                                      nll(input),
                                      givens={input: dataset[no_batch * batch_size:(no_batch + 1) * batch_size]},
                                      name="NLL")

        def _average_nll():
            nlls = []
            for i in range(nb_batches):
                nlls.append(compute_nll(i))

            nlls = np.concatenate(nlls)
            return nlls.mean()
            #return round(nlls.mean(), 6), round(nlls.std() / np.sqrt(nlls.shape[0]), 6)

        super(AverageNLL, self).__init__(_average_nll)


class EarlyStopping(Task, StoppingCriterion):
    def __init__(self, objective, lookahead, save_task=None, eps=0.):
        super(EarlyStopping, self).__init__()

        self.objective = objective
        self.lookahead = lookahead
        self.save_task = save_task
        self.eps = eps
        self.stopping = False

    def check(self, status):
        return status.current_epoch - status.extra['best_epoch'] > self.lookahead

    def init(self, status):
        if 'best_epoch' not in status.extra:
            status.extra['best_epoch'] = 0

        if 'best_objective' not in status.extra:
            status.extra['best_objective'] = float(np.inf)

    def post_epoch(self, status):
        objective = self.objective.view(status)
        if objective + self.eps < status.extra['best_objective']:
            status.extra['best_objective'] = float(objective)
            status.extra['best_epoch'] = status.current_epoch

            if self.save_task is not None:
                self.save_task.execute(status)


class SaveTraining(Task):
    def __init__(self, trainer, savedir, each_epoch=1):
        super(SaveTraining, self).__init__()

        self.savedir = savedir
        self.trainer = trainer
        self.each_epoch = each_epoch

        if not os.path.isdir(self.savedir):
            os.makedirs(self.savedir)

    def execute(self, status):
        self.trainer.save(self.savedir)

    def post_epoch(self, status):
        if status.current_epoch % self.each_epoch == 0:
            self.execute(status)
