from __future__ import division
from __future__ import print_function

import codecs
import random
import numpy as np
import IPython
from itertools import product
from matplotlib import pyplot as plt
import scipy.io
import time

from amazon_utils import *

try:
    import cPickle as pickle
except ImportError:
    import pickle

from amazon_experiment_specifications import DATA_PATH, MODEL_PATH, N_CPUS, n_folds, dim_ll, datasets
from functions.dpp import DPP

# for bookkeeping
ll_flid = {}
ll_dpp_em = {}
ll_dpp_picard = {}
ll_modular = {}

time_flid = {}
time_dpp_em = {}
time_dpp_picard = {}
time_modular = {}

n_items = {}

if __name__ == '__main__':
    # set up data structures for bookkeeping
    for dataset in datasets:
        ll_flid[dataset] = []
        ll_dpp_em[dataset] = []
        ll_dpp_picard[dataset] = []
        ll_modular[dataset] = []

        time_flid[dataset] = []
        time_dpp_em[dataset] = []
        time_dpp_picard[dataset] = []
        time_modular[dataset] = []

        n_items[dataset] = get_amazon_n_items(dataset, data_path=DATA_PATH)

    # Evaluate models
    for dataset, fold, dim in product(datasets, range(n_folds), [dim_ll]):
        print('-' * 30)
        print('dataset: %s (fold %d)' % (dataset, fold + 1))

        data = load_amazon_data(dataset, fold, "train", data_path=DATA_PATH)
        data_test = load_amazon_data(dataset, fold, "test", data_path=DATA_PATH)

        # TODO shift to amazon_utils
        result_flid_f = '{0}/{1}_flid_d_{2}_fold_{3}.pkl'.format(MODEL_PATH, dataset, dim, fold + 1)
        result_mod_f = '{0}/{1}_mod_fold_{2}.pkl'.format(MODEL_PATH, dataset, fold + 1)
        results_model = pickle.load(open(result_flid_f, 'rb'))
        results_modular = pickle.load(open(result_mod_f, 'rb'))

        flid = results_model['model']
        _time_flid = results_model['time']

        modular = results_modular['model']
        _time_modular = results_modular['time']

        # bookkeeping flid
        test = -flid._estimate_LL_exact(data_test)
        ll_flid[dataset].append(test)
        time_flid[dataset].append(_time_flid)

        # bookkeeping modular
        ll_modular[dataset].append(-modular._estimate_LL(data_test))
        time_modular[dataset].append(_time_modular)

        # bookkeeping dpp_em
        K = load_dpp_kernel(dataset, fold, "em", model_path=MODEL_PATH)
        dpp_em = DPP(list(range(K.shape[0])), K)
        ll_dpp_em[dataset].append(-dpp_em._estimate_LL(data_test))
        time_dpp_em[dataset].append(load_dpp_runtime(dataset, fold, "em", model_path=MODEL_PATH))

        # bookkeeping dpp_picard
        K = load_dpp_kernel(dataset, fold, "picard", model_path=MODEL_PATH)
        dpp_picard = DPP(list(range(K.shape[0])), K)
        ll_dpp_picard[dataset].append(-dpp_picard._estimate_LL(data_test))
        time_dpp_picard[dataset].append(load_dpp_runtime(dataset, fold, "picard", model_path=MODEL_PATH))

    # ---- Plotting ----

    print("Plotting.")
    print(ll_flid)

    from matplotlib import rc
    rc('font', **{'family': 'serif', 'serif': ['Times']})
    rc('text', usetex=True)
    rc('font', size=9)
    rc('legend', fontsize=7)

    for plot_type in ['ll', 'rt']:
        means_flid = []
        err_flid = []
        means_dpp_em = []
        err_dpp_em = []
        means_picard_dpp = []
        err_dpp_picard = []
        yt = []
        yt2 = []
        xticks = []
        xtick_labels = []

        if plot_type == 'll':  # log-likelihood plot
            # xlabel = 'Normalized nLL'
            xlabel = 'Relative test log-likelihood differences'
            filename = 'amazon_ll.pdf'
            max_x = 15
            # max_x = 12.5
            loc = 4  # legend location: upper right (=1)

            for dataset in datasets:
                # computes the normalized improvement over modular
                norm_impr = lambda x: - 100 * (np.array(x) - np.array(ll_modular[dataset])) / np.abs(np.array(ll_modular[dataset]))

                impr_flid = norm_impr(ll_flid[dataset])
                impr_dpp_em = norm_impr(ll_dpp_em[dataset])
                impr_dpp_picard = norm_impr(ll_dpp_picard[dataset])

                means_flid.append(np.mean(impr_flid))
                means_dpp_em.append(np.mean(impr_dpp_em))
                means_picard_dpp.append(np.mean(impr_dpp_picard))

                err_flid.append(np.std(impr_flid))
                err_dpp_em.append(np.std(impr_dpp_em))
                err_dpp_picard.append(np.std(impr_dpp_picard))

                yt.append('%s\n\\tiny(N=%d)' % (dataset, n_items[dataset]))
            print(means_flid)
        elif plot_type == 'rt':  # runtime plot
            xlabel = 'Log of runtime in seconds'
            filename = 'amazon_rt.pdf'
            max_x = 3 # 1700
            loc = 1  # legend location: lower right (= 4)

            for dataset in datasets:
                means_flid.append(np.log10(np.sum(time_flid[dataset])))
                err_flid.append(np.std(time_flid[dataset]))

                means_dpp_em.append(np.log10(np.sum(time_dpp_em[dataset])))
                err_dpp_em.append(np.std(time_dpp_em[dataset]))

                means_picard_dpp.append(np.log10(np.sum(time_dpp_picard[dataset])))
                err_dpp_picard.append(np.std(time_dpp_picard[dataset]))

                yt.append(dataset)
                yt2.append('$N=%d$' % (n_items[dataset]))

                # logarithmic axis thing
                xticks = np.linspace(0, max_x, 4)
                for tick in xticks:
                    xtick_labels.append("$10^{%d}$" % (tick))
            print("time NCE: ", means_flid)
            print("time DPP: ", means_dpp_em)
            print("time DPP: ", means_picard_dpp, " (PICARD)")
        else:
            assert False

        plt.figure(figsize=(4 / 2.54, 10 / 2.54))

        index = np.arange(len(datasets))
        print("IDX:", index)
        bar_width = 0.25

        opacity = 0.5
        error_config = {'ecolor': '0.3'}

        for k in range(3):
            if k == 2:
                rects1 = plt.barh(
                         index, means_flid, bar_width,
                         alpha=opacity,
                         color='b',
                         error_kw=error_config,
                         label='FLID',
                         linewidth=0) #,
                         #xerr=err_flid)
                if plot_type == 'll':
                    plt.errorbar(means_flid, index + 0.5 * bar_width,
                        capsize = 1, ecolor = '#666666', fmt=None,
                        xerr=1*np.array(err_flid))

            if k == 0:
                rects1 = plt.barh(
                         index + 2*bar_width, means_picard_dpp, bar_width,
                         alpha=opacity,
                         color='g',
                         error_kw=error_config,
                         label='DPP (FP)',
                         linewidth=0)#,
                         #xerr=err_dpp_picard)
                if plot_type == 'll':
                    plt.errorbar(means_picard_dpp, index + 2.5 * bar_width,
                        capsize = 1, ecolor = '#666666', fmt=None,
                        xerr=1*np.array(err_dpp_picard))


            if k == 1:
                print(means_dpp_em)
                rects2 = plt.barh(
                         index + bar_width, means_dpp_em, bar_width,
                         alpha=opacity,
                         color='r',
                         error_kw=error_config,
                         label='DPP (EM)',
                         linewidth=0)
                         #xerr=err_dpp_em)
                if plot_type == 'll':
                    plt.errorbar(means_dpp_em, index + 1.5 * bar_width,
                        capsize = 1, ecolor = '#666666', fmt=None,
                        xerr=1*np.array(err_dpp_em))

            # set y-ticks
            # plt.yticks(index + 1.5*bar_width, yt)
            plt.yticks([])
            if plot_type == 'll':
                for idx, dataset in enumerate(datasets):
                    xp = -0.6
                    yp = idx + .45
                    plt.gca().text(xp, yp, "%s" % dataset, fontsize=9, ha='right', va='center')
                    plt.gca().text(xp, yp - .35, "\\scriptsize N=%d" % n_items[dataset], fontsize=9, ha='right', va='center')
            plt.xticks(list(range(0, 15, 2)))

            if plot_type == 'rt':
                plt.xticks(xticks, xtick_labels)

            # remove ticks on y-axis
            ax = plt.gca()
            ax.tick_params(axis='y', which='both', length=0)

            # plt.tight_layout()
            plt.axis([0, max_x, -bar_width, len(datasets),])
            if plot_type == 'll':
                plt.legend(loc=loc, prop={'size': 6})

            if plot_type == 'need_an_axis_on_the_right':
                ax2 = ax.twinx()
                ax2.yaxis.set_ticks(index + 1.5*bar_width)
                ax2.yaxis.set_ticklabels(yt2)
                ax2.set_ylim([-bar_width, len(datasets)])
                ax2.tick_params(axis='y', which='both', length=0)

            plt.subplots_adjust(left=0.2)
            # plt.subplots_adjust(bottom=0.1)
            # plt.show()
            if k == 2:
                # plt.savefig("registries%d.pdf" % k)
                plt.savefig(filename, bbox_inches='tight')

