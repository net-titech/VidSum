## Overview


## Usage

To reproduce the results for the amazon data from the paper proceed as follows:

* To run the experiments comparing the likelihood and runtime of FLID to DPPs follow the following steps:
  1. Train the FLID models for all the datasets using ```python amazon_train_models.py```.
  2. Run ```python amazon_ll_and_rt.py``` to evaluate the learned models and compare them to DPPs. This creates two plots (*amazon_ll.pdf* and *amazon_rt.pdf*) showing the likelihood and the runtime comparsion, respectively.
* To run the product recommendation experiment follow the following steps:
  1. Generate test sets and baseline results by calling ```python generate_ranking_task.py```,
  2. Perform ranking using FLID by calling ```python amazon_ranking.py```.
  3. Evaluate the ranking results by calling ```python ranking.py```. This will create a plot named *amazon_mrr.pdf* showing a performance comparison of the modular model, DPPs and FLID.

Pretrained DPPs.

## Citation

If you use this code for publications, please cite our AISATA paper:

    @inproceedings{tschiatschek16learning,
	Author = {Sebastian Tschiatschek and Josip Djolonga and Andreas Krause},
	Booktitle = {Proc. International Conference on Artificial Intelligence and Statistics (AISTATS)},
	Month = {May},
	Title = {Learning Probabilistic Submodular Diversity Models Via Noise Contrastive Estimation},
	Year = {2016}}

