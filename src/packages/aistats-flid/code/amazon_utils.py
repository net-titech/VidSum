import scipy.io
import os
import os.path
import h5py
import ujson

def save_to_csv(filename, lst):
    with open(filename, "wt") as f:
        for S in lst:
            for i, k in enumerate(S):
                f.write("%d" % k)
                if i < len(S) - 1:
                    f.write(",")
            f.write("\n")

def create_dir_if_not_exists(d):
    if not os.path.exists(d):
        os.makedirs(d)

def load_amazon_names(filename):
    with codecs.open(filename, 'rb') as f:
        return [name.decode('unicode_escape') for name in f]


def load_amazon_data(dataset, fold, type, data_path):
    assert type == "train" or type == "test"
    assert os.path.exists(data_path), "Path {0} not found".format(data_path)
    filename = os.path.join(data_path,
            '1_100_100_100_{0}_regs_{1}_fold_{2}.csv'.format(
            dataset, type, fold + 1))
    return _load_amazon_data(filename)


def _load_amazon_data(filename):
    with open(filename, 'r') as f:
        return [[int(x) - 1 for x in line.strip().split(',')]
            for line in f if line.strip()]

def get_amazon_n_items(dataset, data_path):
    assert os.path.exists(data_path), "Path {0} not found".format(data_path)
    fold = 1  # the actual fold does not matter
    data_train = load_amazon_data(dataset, fold, "train", data_path=data_path)
    data_test = load_amazon_data(dataset, fold, "test", data_path=data_path)
    return max([max(max(x) for x in data_train), max(max(x) for x in data_test)]) + 1

def load_amazon_ranking_data(filename):
    """
    Only difference to amazon_data is that item indices must not be corrected.
    """
    with open(filename, 'r') as f:
        return ujson.load(f)
    # with open(filename, 'r') as f:
    #     return [[int(x) for x in line.strip().split(',')]
    #            for line in f if line.strip()]

def load_amazon_ranking_data_dpp(filename):
    """
    Only difference to amazon_data is that item indices must not be corrected.
    """
    with open(filename, 'r') as f:
        return [[int(x) for x in line.strip().split(',')]
              for line in f if line.strip()]

def load_dpp_kernel(dataset, fold, mode, model_path):
    assert mode == "em" or mode == "picard"
    if mode == "em":
        filename = os.path.join(model_path, "{0}_dpp_em_fold_{1}.h5".format(
            dataset, fold + 1))
    elif mode == "picard":
        filename = os.path.join(model_path, "{0}_dpp_picard_fold_{1}.h5".format(
            dataset, fold + 1))
    assert os.path.exists(filename), "Kernel file {0} not found.".format(filename)

    f = h5py.File(filename, 'r')
    K = f['kernel'].value
    f.close()

    return K

def load_dpp_runtime(dataset, fold, mode, model_path):
    assert mode == "em" or mode == "picard"
    if mode == "em":
        filename = os.path.join(model_path, "{0}_dpp_em_fold_{1}.h5".format(
            dataset, fold + 1))
    elif mode == "picard":
        filename = os.path.join(model_path, "{0}_dpp_picard_fold_{1}.h5".format(
            dataset, fold + 1))
    assert os.path.exists(filename), "Data file {0} not found.".format(filename)

    f = h5py.File(filename, 'r')
    rt = f['runtime'].value
    f.close()

    return rt


# TODO REMOVE
def load_dpp_result(dataset, fold, RESULT_PATH):
    model_f = '{0}/{1}_fold_{2}.mat'.format(
            RESULT_PATH, dataset, fold + 1)
    print("Loading matlab model from %s." % (model_f))
    mat = scipy.io.loadmat(model_f)
    ll_em = mat['ll_test_em'][0][0]
    rt_em = mat['rt_em'][0][0]
    ll_picard = mat['ll_test_picard'][0][0]
    rt_picard = mat['rt_picard'][0][0]


    return {'em': (-ll_em, rt_em), 'picard': (-ll_picard, rt_picard)}

