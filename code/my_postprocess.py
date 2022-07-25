import pickle
import pandas as pd

result_path = '../result/pmf-result.pkl'
save_path = '../result/pmf-result.xlsx'

f = open(result_path, 'rb')
results = pickle.load(f)
print(results.keys())


with pd.ExcelWriter(save_path) as writer:
    for k in results.keys():
        pd.DataFrame(results[k]).to_excel(writer, sheet_name=k, index=False)
    
    writer.save()
    
f.close()

