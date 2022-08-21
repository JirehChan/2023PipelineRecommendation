import os
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset_name')
parser.add_argument('-s', '--save_name')
parser.add_argument('-p', '--part_time')
args = parser.parse_args()

train_accs = {}
val_accs = {}
test_accs = {}

regrets = {}
ndcgs = {}
evaled = {}
topks = {}

warm_train_losses = {}
warm_val_losses = {}

for i in range(int(args.part_time)):
    if i==10:
        continue
    print(i, 'trainaccs')
    file_name = '../result/{}/{}/trainaccs-{}.pkl'.format(args.dataset_name, args.save_name, i)
    with open(file_name, 'rb') as f:
        train_accs.update(pickle.load(f))
    #os.remove(file_name)

    print(i, 'valaccs')
    file_name = '../result/{}/{}/valaccs-{}.pkl'.format(args.dataset_name, args.save_name, i)
    with open(file_name, 'rb') as f:
        val_accs.update(pickle.load(f))
    #os.remove(file_name)

    print(i, 'testaccs')
    file_name = '../result/{}/{}/testaccs-{}.pkl'.format(args.dataset_name, args.save_name, i)
    with open(file_name, 'rb') as f:
        test_accs.update(pickle.load(f))
    #os.remove(file_name)
    
    print(i, 'regrets')
    file_name = '../result/{}/{}/regrets-{}.pkl'.format(args.dataset_name, args.save_name, i)
    with open(file_name, 'rb') as f:
        regrets.update(pickle.load(f))
    #os.remove(file_name)
    
    print(i, 'ndcgs')
    file_name = '../result/{}/{}/ndcgs-{}.pkl'.format(args.dataset_name, args.save_name, i)
    with open(file_name, 'rb') as f:
        ndcgs.update(pickle.load(f))
    #os.remove(file_name)
    
    print(i, 'evaled')
    file_name = '../result/{}/{}/evaled-{}.pkl'.format(args.dataset_name, args.save_name, i)
    with open(file_name, 'rb') as f:
        evaled.update(pickle.load(f))
    #os.remove(file_name)
    
    print(i, 'topks')
    file_name = '../result/{}/{}/topks-{}.pkl'.format(args.dataset_name, args.save_name, i)
    with open(file_name, 'rb') as f:
        topks.update(pickle.load(f))
    #os.remove(file_name)

    print(i, 'warm_train_losses')
    file_name = '../result/{}/{}/warm_train_losses-{}.pkl'.format(args.dataset_name, args.save_name, i)
    with open(file_name, 'rb') as f:
        warm_train_losses.update(pickle.load(f))
    #os.remove(file_name)

    print(i, 'warm_val_losses')
    file_name = '../result/{}/{}/warm_val_losses-{}.pkl'.format(args.dataset_name, args.save_name, i)
    with open(file_name, 'rb') as f:
        warm_train_losses.update(pickle.load(f))
    #os.remove(file_name)
''''''
f = open('../result/{}/{}/trainaccs.pkl'.format(args.dataset_name, args.save_name),'wb')
pickle.dump(train_accs,f)
f.close()

f = open('../result/{}/{}/valaccs.pkl'.format(args.dataset_name, args.save_name),'wb')
pickle.dump(val_accs,f)
f.close()

f = open('../result/{}/{}/testaccs.pkl'.format(args.dataset_name, args.save_name),'wb')
pickle.dump(test_accs,f)
f.close()

f = open('../result/{}/{}/regrets.pkl'.format(args.dataset_name, args.save_name),'wb')
pickle.dump(regrets,f)
f.close()

f = open('../result/{}/{}/evaled.pkl'.format(args.dataset_name, args.save_name),'wb')
pickle.dump(evaled, f)
f.close()

f = open('../result/{}/{}/ndcgs.pkl'.format(args.dataset_name, args.save_name),'wb')
pickle.dump(ndcgs,f)
f.close()

f = open('../result/{}/{}/topks.pkl'.format(args.dataset_name, args.save_name),'wb')
pickle.dump(topks,f)
f.close()    

f = open('../result/{}/{}/warm_train_losses.pkl'.format(args.dataset_name, args.save_name),'wb')
pickle.dump(warm_train_losses,f)
f.close() 

f = open('../result/{}/{}/warm_val_losses.pkl'.format(args.dataset_name, args.save_name),'wb')
pickle.dump(warm_val_losses,f)
f.close() 

'''
for i in range(int(args.part_time)):
    file_name = '../result/{}/{}/trainaccs-{}.pkl'.format(args.dataset_name, args.save_name, i)
    os.remove(file_name)

    file_name = '../result/{}/{}/valaccs-{}.pkl'.format(args.dataset_name, args.save_name, i)
    os.remove(file_name)

    file_name = '../result/{}/{}/testaccs-{}.pkl'.format(args.dataset_name, args.save_name, i)
    os.remove(file_name)
    
    file_name = '../result/{}/{}/regrets-{}.pkl'.format(args.dataset_name, args.save_name, i)
    os.remove(file_name)
    
    file_name = '../result/{}/{}/ndcgs-{}.pkl'.format(args.dataset_name, args.save_name, i)
    os.remove(file_name)
    
    file_name = '../result/{}/{}/evaled-{}.pkl'.format(args.dataset_name, args.save_name, i)
    os.remove(file_name)
    
    file_name = '../result/{}/{}/topks-{}.pkl'.format(args.dataset_name, args.save_name, i)
    os.remove(file_name)

    file_name = '../result/{}/{}/warm_train_losses-{}.pkl'.format(args.dataset_name, args.save_name, i)
    os.remove(file_name)
    file_name = '../result/{}/{}/warm_val_losses-{}.pkl'.format(args.dataset_name, args.save_name, i)
    os.remove(file_name)
'''