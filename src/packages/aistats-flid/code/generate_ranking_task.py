"""
Generates the data for the ranking task. For every S in the test set with
at least two items and less than MAX_LEN items it generates new baskets (by
removing a single item). The task is to predict the removed item.
"""

from __future__ import division
from __future__ import print_function

import codecs
import random
import numpy as np
from itertools import product, combinations
import ujson

from amazon_utils import *

from functions.modular import ModularFun

## Generates the data for the ranking task

try:
    import cPickle as pickle
except ImportError:
    import pickle

from amazon_experiment_specifications import DATA_PATH, RANKING_DATA_PATH, MODEL_PATH, N_CPUS, n_folds, datasets, MAX_LEN

def get_proposal(f, Sorig):
    """
    Get the proposal for an additional item from f, given Sorig
    """
    assert isinstance(f, ModularFun), "Function must be modular."

    utils = np.copy(f.s)
    utils[Sorig] = - float('inf')
    t = np.argsort(utils)
    t = t[len(Sorig):]
    return t[::-1]

if __name__ == '__main__':
    create_dir_if_not_exists(RANKING_DATA_PATH)

    for dataset, fold in product(datasets, range(0, n_folds)):
        np.random.seed(20150820)

        print('-' * 30)
        print('dataset: %s (fold %d)' % (dataset, fold + 1))

        # file names
        result_mod_f = '{0}/{1}_mod_fold_{2}.pkl'.format(MODEL_PATH, dataset, fold + 1)
        result_ranking_gt_f = '{0}/{1}_gt_fold_{2}.json'.format(RANKING_DATA_PATH, dataset, fold + 1)
        result_ranking_partial_f = '{0}/{1}_partial_fold_{2}.json'.format(RANKING_DATA_PATH, dataset, fold + 1)
        result_ranking_mod_f = '{0}/{1}_mod_fold_{2}.json'.format(RANKING_DATA_PATH, dataset, fold + 1)

        data_test = load_amazon_data(dataset, fold, "test", data_path=DATA_PATH)

        # Load modular model
        results_modular = pickle.load(open(result_mod_f, 'rb'))
        modular = results_modular['model']
        V = modular.V

        list_prop_modular = []
        list_gt = []
        list_partial = []  # keep track of the partial shown sets (as interface for Matlab)
        for i, S in enumerate(data_test):
            if len(S) < 2 or len(S) > MAX_LEN:  # only considers sets S with 1 < |S| <= MAX_LEN
                continue

            for n_show in [len(S) - 1]:
                for A in combinations(list(range(len(S))), n_show):
                    # new basket
                    S_new = np.array(S)[list(A)]
                    S_new = S_new.tolist()

                    # ground truth
                    gt = np.array(S)
                    gt = np.delete(gt, list(A)).tolist()
                    # TODO remove? gt = np.array(S)[list(set(range(len(S))).difference(list(A)))]
                    # TODO remove? gt = gt.tolist()

                    # get proposal from modular distribution
                    prop_modular = get_proposal(modular, S_new[:])

                    # bookkeeping
                    list_prop_modular.append(prop_modular.tolist())
                    list_gt.append(gt)
                    list_partial.append(S_new)

        ujson.dump(list_prop_modular, open(result_ranking_mod_f, 'w'))
        ujson.dump(list_gt, open(result_ranking_gt_f, 'w'))
        ujson.dump(list_partial, open(result_ranking_partial_f, 'w'))


