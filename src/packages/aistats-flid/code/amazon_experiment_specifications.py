import multiprocessing

DATA_PATH = 'amazon_data'
MODEL_PATH = 'models'
RANKING_DATA_PATH = 'ranking_data'
N_CPUS = multiprocessing.cpu_count()  # How many processors to parallelize the sampling on.
MAX_LEN = 100  # max length of a sample to be considered for ranking

n_folds = 10

F_NOISE_RANGE = [10, 10]  # Number of noise samples = F_NOISE * (number of training samples); uses F_NOISE=5 for dim=2, and F_NOISE=10 for dim=10
dim_range = [3, 10] # train models for this number of latent dimensions
lr_range = [1e0, 1e0]
n_iter_range = [20, 20]

dim_ll = 3  # number of latent dimensions for likelihood computation
dim_ranking = 10  # number of latent dimensions for ranking

datasets = ['safety', 'furniture', 'carseats', 'strollers',
            'health', 'bath', 'media', 'toys',
            'bedding', 'apparel', 'diaper', 'gear',
            'feeding']
datasets = ['safety', 'furniture', 'carseats', 'strollers']
# datasets = ['feeding']
datasets = datasets[::-1]

