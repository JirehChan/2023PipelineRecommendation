# ReadMe

## code

- bo.py, gplvm.py, kernels.py, plotting.py, utils.py: the code for PMF method (from https://github.com/rsheth80/pmf-automl)
- my_main.py: the main run code
- my_recommends.py: whole the pipeline recommender implement code
- my_tool.py: the code for load dataset, write log, bayesian process, etc.
- my_postprocessing.py: the code for the final run

## data

- data for 110-classifiers and openml dataset
- dataset_feats.csv: the dataset features
- performance.csv: the performance
- pipelines.json: pipelines information

## settings

- example of warm_starters settings

## run example

```
python my_main.py --dataset_name openml --random_seed 0 --save_path 0103-80%-s0 --warm_path 0103-80% --bo_n_init 5 --bo_n_iters 100 --is_bayes 0 --nan_ratio 0.8 --part_name 0 --is_bayes 1 --model_path  0103-80%-s0 --data_path 0103-80%-s0
```

- dataset_name
- random_seed
- warm_path: the path to save warm starter settings, example in /settings/0103-80%/warm_starters-0.txt
- is_bayes: wether use the bayesian process
- bo_n_init: the number of warm stater iteration
- bo_n_iters: the whole number of running iteration
- nan_ratio: the sparisty ratio setting
- part_name: the number n of warm_starters-n.txt when have multiple warm_starters settings
- model_path: used when have already trained the model
- data_path: used when have already split the dataset to train/val/test