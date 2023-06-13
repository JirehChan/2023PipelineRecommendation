import pickle
import pandas as pd
import my_tool as mt

args = mt.parse_arg()

results = {}

for i in range(int(args.part_name)):
    f = open('../result/{}/{}/result-{}.pkl'.format(args.dataset_name, args.save_path, i), 'rb')
    res = pickle.load(f)

    for k in res.keys():
        results[k] = res[k]

    f.close()

f = open('../result/{}/{}/result.pkl'.format(args.dataset_name, args.save_path, i), 'wb')
pickle.dump(results, f)
f.close()

f = open('../result/{}/{}/result.pkl'.format(args.dataset_name, args.save_path, i), 'rb')
results = pickle.load(f)
f.close()

keys = list(results.keys())
res_keys = []
train_accs = []
val_accs = []
test_accs = []
bo_accs = []

for k in keys:
    if args.is_bayes==1:
        bo_accs.append(results[k]['bo_accs'].mean(axis=1).tolist())
    if k == 'random1x':
        continue
    res_keys.append(k)
    train_accs.append(results[k]['train_best_accs'].mean(axis=0).tolist())
    val_accs.append(results[k]['val_best_accs'].mean(axis=0).tolist())
    test_accs.append(results[k]['test_best_accs'].mean(axis=0).tolist())

train_df = pd.DataFrame(train_accs, index=res_keys)
val_df = pd.DataFrame(val_accs, index=res_keys)
test_df = pd.DataFrame(test_accs, index=res_keys)
if args.is_bayes==1:
    bo_df = pd.DataFrame(bo_accs, index = keys)

train_df.to_csv('../result/{}/{}/train.csv'.format(args.dataset_name, args.save_path, i))
val_df.to_csv('../result/{}/{}/valdation.csv'.format(args.dataset_name, args.save_path, i))
test_df.to_csv('../result/{}/{}/test.csv'.format(args.dataset_name, args.save_path, i))

import re 
baseline_index = []
pmm_index = []
bpmm_index = []

for ix in train_df.index:
    a = ix.replace(' ', '')

    if re.match(r'pmm(\S*)\)', a):
        pmm_index.append(ix)
    elif re.match(r'bpmm(\S*)\)', a):
        bpmm_index.append(ix)
    else:
        baseline_index.append(ix)

indexs = baseline_index + pmm_index + bpmm_index
tr_df = train_df.loc[indexs].iloc[:,:5]
va_df = val_df.loc[indexs].iloc[:,:5]
te_df = test_df.loc[indexs].iloc[:,:5]
pd.concat([tr_df, va_df, te_df], axis=1).to_csv('../result/{}/{}/res.csv'.format(args.dataset_name, args.save_path, i))

if args.is_bayes==1:
    indexs = ['random1x']+indexs
    bo_df = bo_df.loc[indexs]
    bo_df.to_csv('../result/{}/{}/bo.csv'.format(args.dataset_name, args.save_path, i))