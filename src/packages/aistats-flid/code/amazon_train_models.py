from __future__ import division
from __future__ import print_function

import codecs
import random
import numpy as np
from itertools import product
import time

from fast_train import Trainer
from functions.flid import DiversityFun
from functions.modular import ModularFun
from amazon_utils import *
from multiprocessing import Pool

try:
    import cPickle as pickle
except ImportError:
    import pickle

from amazon_experiment_specifications import DATA_PATH, MODEL_PATH, F_NOISE_RANGE, N_CPUS, n_folds, dim_range, datasets, lr_range, n_iter_range

def sample_once(f_noise):
    print("N_SAMPLES = ", N_SAMPLES)
    return f_noise.sample(N_SAMPLES // N_CPUS)

def train_model(data, n_items, lr, n_iter):
    global N_SAMPLES

    start = time.time()
    # Count marginals
    item_marg = np.zeros(n_items) + 1 # pseudo count
    for sample in data:
        item_marg[sample] += 1
    item_marg /= len(data)
    s = -np.log(1. / item_marg - 1.)  # The modular term

    f_noise = ModularFun(list(range(n_items)), np.copy(s))
    end = time.time()
    time_modular = end - start

    N_SAMPLES = F_NOISE * len(data)

    # generate noise samples
    print('Sampling...')
    pool = Pool(N_CPUS)

    samples_all = pool.map(sample_once, [f_noise] * N_CPUS)
    data_noise = []
    for samples in samples_all:
        data_noise.extend(samples)

    pool.close()
    pool.join()
    print("... done.")

    # TODO Get rid of this # Remove empty sets.
    # data_noise = list(filter(lambda x: len(x) > 0, data_noise))

    f_model = DiversityFun(list(range(n_items)), dim)
    f_model.utilities = np.copy(f_noise.s)
    f_model.n_logz[0] = -f_noise.logz;

    start = time.time()
    trainer = Trainer(data, data_noise, f_noise.s, n_items=n_items, dim=dim)
    trainer.train(n_iter, lr, 0.1, plot=False)
    f_model.W = trainer.weights
    f_model.utilities = trainer.unaries
    f_model.n_logz = trainer.n_logz
    end = time.time()
    print("TRAINING TOOK ", (end - start))

    end = time.time()
    time_nce = end - start

    return f_model, f_noise, time_nce, time_modular


if __name__ == '__main__':
    create_dir_if_not_exists(MODEL_PATH)

    global F_NOISE
    for dataset, fold, pars in product(datasets, range(n_folds), zip(dim_range, F_NOISE_RANGE, lr_range, n_iter_range)):
        dim, F_NOISE, lr, n_iter = pars
        print('-' * 30)
        print('dataset: %s (fold %d)' % (dataset, fold + 1))
        print("--> dims = %d" % dim)

        np.random.seed(20150820)  # fix random seed

        result_flid_f = '{0}/{1}_flid_d_{2}_fold_{3}.pkl'.format(MODEL_PATH, dataset, dim, fold + 1)
        result_mod_f = '{0}/{1}_mod_fold_{2}.pkl'.format(MODEL_PATH, dataset, fold + 1)

        data = load_amazon_data(dataset, fold, "train", data_path=DATA_PATH)
        random.shuffle(data)

        n_items = get_amazon_n_items(dataset, data_path=DATA_PATH)

        f_model, f_noise, time_logflid_, time_modular_ = train_model(data, n_items=n_items, lr=lr, n_iter=n_iter)
        results_model = {'model': f_model, 'time': time_logflid_}
        results_modular = {'model': f_noise, 'time': time_modular_}

        # dump results
        pickle.dump(results_model, open(result_flid_f, 'wb'))
        pickle.dump(results_modular, open(result_mod_f, 'wb'))


