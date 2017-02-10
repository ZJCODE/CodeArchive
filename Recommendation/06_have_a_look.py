import pandas as pd
from math import log
import numpy as np
model_pred = pd.read_csv('./Data/test_pred_loc.csv')
baseline_pred = pd.read_csv('./Data/baseline_pred_loc.csv')

model_hr = []
baseline_hr = []

for n in range(1,11):
    model_pred_n  =  [1 if x <=n else 0 for x in model_pred.location.values]
    baseline_pred_n = [1 if x <=n else 0 for x in baseline_pred.location.values]
    print sum(model_pred_n)
    print sum(baseline_pred_n)
    print "--------------"
    model_hr.append(sum(model_pred_n) * 1.0  / len(model_pred))
    baseline_hr.append(sum(baseline_pred_n) * 1.0 / len(baseline_pred))


print 'Hit'
print '---------------------'
print model_hr

print baseline_hr

print 'NDCG'

model_NDCG_at10 = []
baseline_NDCG_at10 = []

for p in model_pred.location.values:
	if p <= 10 :
		model_NDCG_at10.append(1.0/(log(1+p) / log(2)))
	else:
		model_NDCG_at10.append(0)

model_NDCG = np.mean(model_NDCG_at10)

print 'model_NDCG'
print model_NDCG

for p in baseline_pred.location.values:
	if p <= 10 :
		baseline_NDCG_at10.append(1.0/(log(1+p) / log(2)))
	else:
		baseline_NDCG_at10.append(0)

baseline_NDCG = np.mean(baseline_NDCG_at10)


print 'baseline_NDCG'
print baseline_NDCG

 
	

