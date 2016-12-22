from __future__ import division, print_function
from matplotlib import pyplot as plt
import numpy as np
import IPython
from itertools import product
from itertools import chain, combinations
import ujson
import pickle
from amazon_utils import load_amazon_ranking_data_dpp

from amazon_experiment_specifications import datasets, n_folds, dim_ranking, MODEL_PATH, RANKING_DATA_PATH

import multiprocessing
N_CPUS = multiprocessing.cpu_count()

plot = True

def compute_acc_and_rank(true_set, suggested):
    """
    Computes the accuracy and the rank of suggested, when the correct
    prediction would be true_set. (Note that the function is specialized to
    true_set containing only a single item.)
    """
    assert len(true_set) == 1

    accuracy = 1. if suggested[0] in true_set else 0.
    rank = 1. + suggested.index(true_set[0])

    return (accuracy, rank)

if __name__ == '__main__':
    # setup data structures for bookkeeping
    results_all_flid = dict()
    results_all_mod = dict()
    results_all_dpp = dict()
    rank_flid = dict()
    rank_mod = dict()
    rank_dpp = dict()
    for dataset in datasets:
        results_all_flid[dataset] = []
        results_all_mod[dataset] = []
        results_all_dpp[dataset] = []

        rank_flid[dataset] = []
        rank_mod[dataset] = []
        rank_dpp[dataset] = []

    for dataset, fold, dim in product(datasets, range(n_folds), [dim_ranking]):
        print('-' * 30)
        print('dataset: %s (fold %d)' % (dataset, fold + 1))
        print('dim=%d' % dim)

        result_ranking_flid_f = '{0}/{1}_flid_d_{2}_fold_{3}.json'.format(RANKING_DATA_PATH, dataset, dim, fold + 1)
        result_ranking_mod_f = '{0}/{1}_mod_fold_{2}.json'.format(RANKING_DATA_PATH, dataset, fold + 1)
        result_ranking_dpp_f = '{0}/{1}_dpp_em_fold_{2}.json'.format(RANKING_DATA_PATH, dataset, fold + 1)
        result_ranking_gt_f = '{0}/{1}_gt_fold_{2}.json'.format(RANKING_DATA_PATH, dataset, fold + 1)

        GROUND_TRUTH = result_ranking_gt_f
        if dataset == "all_small":
            METHODS = {
               'flid': result_ranking_flid_f,
               'mod': result_ranking_mod_f,
            }
        else:
            METHODS = {
               'flid': result_ranking_flid_f,
               'mod': result_ranking_mod_f,
               'dpp': result_ranking_dpp_f,
            }


        results = dict()
        with open(GROUND_TRUTH) as f_gt:
            list_gt = ujson.load(f_gt)
            for method, filename in METHODS.items():
                # print('processing', method)
                result = None
                avg_score = []
                avg_rank = []
                with open(filename, 'r') as f_sc:
                    print(filename)
                    list_sc = ujson.load(f_sc)
                    print("Read %d lists." % len(list_sc))
                    assert len(list_gt) == len(list_sc), "Length of ground truth does not match length of predictions."
                    for true_set, suggested in zip(list_gt, list_sc):
                        accuracy, rank = compute_acc_and_rank(true_set, suggested)
                        avg_score.append(accuracy)
                        avg_rank.append(rank)

                if method == 'flid':
                    results_all_flid[dataset].append(np.mean(avg_score))
                    rank_flid[dataset].append(np.mean([1./rank for rank in avg_rank]))
                elif method == 'mod':
                    results_all_mod[dataset].append(np.mean(avg_score))
                    rank_mod[dataset].append(np.mean([1./rank for rank in avg_rank]))
                elif method == 'dpp':
                    results_all_dpp[dataset].append(np.mean(avg_score))
                    rank_dpp[dataset].append(np.mean([1./rank for rank in avg_rank]))
                else:
                    assert False, "Invalid method."

    if not plot:
        print("Not plotting :(")
        import sys
        sys.exit(1)

    import IPython
    IPython.embed()

    from matplotlib import rc
    rc('font', **{'family': 'serif', 'serif': ['Times']})
    rc('text', usetex=True)
    rc('font', size=9)
    rc('legend', fontsize=7)

    for plot_type in ['acc', 'mrr']:
        means_flid = []
        err_flid = []
        means_dpp = []
        err_dpp = []
        means_modular = []
        err_modular = []
        xticks = []
        xtick_labels = []

        if plot_type == 'acc':  # accuracy plot
            filename = 'amazon_acc.pdf'
            max_x = 25

            for dataset in datasets:
                means_modular.append(100 * np.mean(results_all_mod[dataset]))
                means_flid.append(100 * np.mean(results_all_flid[dataset]))
                means_dpp.append(100 * np.mean(results_all_dpp[dataset]))

                err_modular.append(100 * np.std(results_all_mod[dataset]))
                err_flid.append(100 * np.std(results_all_flid[dataset]))
                err_dpp.append(100 * np.std(results_all_dpp[dataset]))
            print(means_flid)
        elif plot_type == 'mrr':  # mean reciprocal ranking
            filename = 'amazon_mrr.pdf'
            max_x = 50 # 1700

            for dataset in datasets:
                means_flid.append(100 * np.mean(rank_flid[dataset]))
                means_dpp.append(100 * np.mean(rank_dpp[dataset]))
                means_modular.append(100 * np.mean(rank_mod[dataset]))

                err_modular.append(100 * np.std(rank_mod[dataset]))
                err_flid.append(100 * np.std(rank_flid[dataset]))
                err_dpp.append(100 * np.std(rank_dpp[dataset]))
            print(means_flid)
        else:
            assert False

        plt.figure(figsize=(4 / 2.54, 10 / 2.54))

        index = np.arange(len(datasets))
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
                         #xerr=err_nce)
                plt.errorbar(means_flid, index + 0.5 * bar_width,
                        capsize = 1, ecolor = '#666666', fmt=None,
                        xerr=1*np.array(err_flid))

            if k == 0:
                rects1 = plt.barh(
                         index + 2*bar_width, means_modular, bar_width,
                         alpha=opacity,
                         color='k',
                         error_kw=error_config,
                         label='modular',
                         linewidth=0)#,
                         #xerr=err_picard_dpp)
                plt.errorbar(means_modular, index + 2.5 * bar_width,
                        capsize = 1, ecolor = '#666666', fmt=None,
                        xerr=1*np.array(err_modular))

            if k == 1:
                rects2 = plt.barh(
                         index + bar_width, means_dpp, bar_width,
                         alpha=opacity,
                         color='r',
                         error_kw=error_config,
                         label='DPP (EM)',
                         linewidth=0)#,
                         #xerr=err_em_dpp)
                plt.errorbar(means_dpp, index + 1.5 * bar_width,
                        capsize = 1, ecolor = '#666666', fmt=None,
                        xerr=1*np.array(err_dpp))


            plt.yticks([])

            # plt.xticks(xticks, xtick_labels)
            if plot_type == 'mrr':
                xrange = list(range(0, 60, 10))
                plt.xticks(xrange)

            # remove ticks on y-axis
            ax = plt.gca()
            ax.tick_params(axis='y', which='both', length=0)

            # plt.tight_layout()
            plt.axis([0, max_x, -bar_width, len(datasets),])

            if plot_type == 'mrr':
                plt.legend(loc=4, prop={'size': 6})

            plt.subplots_adjust(left=0.2)
            # plt.subplots_adjust(bottom=0.1)
            # plt.show()
            if k == 2:
                # plt.savefig("registries%d.pdf" % k)
                plt.savefig(filename, bbox_inches='tight')

