"""
This script loads pre-trained flid models and uses them for ranking. It goes
through the basekets in the test data and proposes a list of elements to be
added (the order of the items corresponds to the likelihood of an item being
added conditionend on that we only want to add one item).
"""
from __future__ import division
from __future__ import print_function

import codecs
import random
import numpy as np
from itertools import product
from scipy.misc import logsumexp as lse
import time
import ujson

from amazon_utils import *
from amazon_experiment_specifications import datasets, n_folds, dim_ranking, MODEL_PATH, RANKING_DATA_PATH
import multiprocessing
from functions.flid import DiversityFun
from functions.dpp import DPP
import itertools

try:
    import cPickle as pickle
except ImportError:
    import pickle

N_CPUS = multiprocessing.cpu_count()

def get_proposal(args, n_propose=1, sample='topN'):
    """
    Get the proposal for an additional item from f, given S
    """
    Sorig, f = args

    assert isinstance(f, DiversityFun), "Model must be an instance of DiversityFun."
    assert n_propose == 1  # modified to work for this case only

    V = f.V
    N = len(V)

    S = Sorig[:]
    # Vred = np.array(list(set(V).difference(set(S))))
    Vred = np.delete(np.array(V), S)
    f_S = f(S)

    probs = []
    gains = []
    vals = f.all_singleton_adds(S)
    vals = np.delete(vals, S)
    gains = vals - f_S
    # for i in Vred:
    #     f_Si = f(list(set(S).union(set([i]))))
    #     probs.append(f_Si)
    #     gains.append(f_Si - f_S)
    probs = np.exp(vals - lse(vals))

    order = Vred[np.argsort(probs)[::-1]]
    probs = np.sort(probs)[::-1]

    return {'order': order, 'probs': probs}

def get_proposal_dpp(args, typ='marginal'):
    S, dpp = args

    if typ == 'marginal':
        return dpp._get_proposal_marginal(S)
    else:
        return dpp._get_proposal(S)

if __name__ == '__main__':
    pool = multiprocessing.Pool(N_CPUS)
    for dataset, fold, dim in product(datasets, range(n_folds), [dim_ranking]):
        print('-' * 30)
        print('dataset: %s (fold %d)' % (dataset, fold + 1))

        # load model
        result_flid_f = '{0}/{1}_flid_d_{2}_fold_{3}.pkl'.format(MODEL_PATH, dataset, dim, fold + 1)
        results_model = pickle.load(open(result_flid_f, 'rb'))
        flid = results_model['model']

        # load data
        result_ranking_gt_f = '{0}/{1}_gt_fold_{2}.json'.format(RANKING_DATA_PATH, dataset, fold + 1)
        result_ranking_partial_f = '{0}/{1}_partial_fold_{2}.json'.format(RANKING_DATA_PATH, dataset, fold + 1)

        data_ranking = load_amazon_ranking_data(result_ranking_partial_f)

        print("Performing ranking.")
        prop_model_dicts = pool.map(get_proposal, zip(data_ranking, itertools.repeat(flid)))

        # extract proposals from list of dictionaries
        prop_model = [d['order'].tolist() for d in prop_model_dicts]

        # save results
        result_ranking_flid_f = '{0}/{1}_flid_d_{2}_fold_{3}.json'.format(RANKING_DATA_PATH, dataset, dim, fold + 1)
        with open(result_ranking_flid_f, 'w') as f:
            ujson.dump(prop_model, f)

    # perform ranking with DPPs
    for dataset, fold in product(datasets, range(n_folds)):
        print('-' * 30)
        print('DPP dataset: %s (fold %d)' % (dataset, fold + 1))

        # load dpp model
        K = load_dpp_kernel(dataset, fold, "em", model_path=MODEL_PATH)
        dpp_em = DPP(list(range(K.shape[0])), K)

        # load data
        result_ranking_gt_f = '{0}/{1}_gt_fold_{2}.json'.format(RANKING_DATA_PATH, dataset, fold + 1)
        result_ranking_partial_f = '{0}/{1}_partial_fold_{2}.json'.format(RANKING_DATA_PATH, dataset, fold + 1)

        data_ranking = load_amazon_ranking_data(result_ranking_partial_f)

        print("Performing ranking.")
        prop_model_dicts = pool.map(get_proposal_dpp, zip(data_ranking, itertools.repeat(dpp_em)))

        # extract proposals from list of dictionaries
        prop_model = [list([int(i) for i in d]) for d in prop_model_dicts]

        # save results
        result_ranking_dpp_f = '{0}/{1}_dpp_em_fold_{2}.json'.format(RANKING_DATA_PATH, dataset, fold + 1)
        with open(result_ranking_dpp_f, 'w') as f:
            ujson.dump(prop_model, f)

    pool.close()
    pool.join()

