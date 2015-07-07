import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score, train_test_split

import sys
sys.path.append('/home/simengy/git/xgboost/wrapper/')
import xgboost as xgb

import datetime

EXPORT = True


start = datetime.datetime.now()

data = pd.read_csv('feature/outdate/all_total_3.csv', header=False)
label = pd.read_csv('../data/train/truth_train.csv', header=False)

assert data.shape[0] == label.shape[0], 'Sizes of features (%d) and labels (%d) do not match!' % (data.shape[0], label.shape[0])
print data.drop('enrollment_id', axis=1).mean()

data = data.fillna(data.mean())

X_train, X_test, Y_train, y_test = train_test_split(
    data, 
    label,
    test_size = 0.0
)


print "Build model:"

param = {'bst:max_depth':10, 'bst:eta':0.009, 'silent':0, 'objective':'binary:logistic', 'subsample':0.6 }
param['nthread'] = 25
plst = param.items()
plst += [('eval_metric', 'auc')]
num_round = 1000
    
dX_base = xgb.DMatrix(X_train.drop('enrollment_id', axis=1).as_matrix(), label = Y_train['label'].as_matrix())

if EXPORT == False:
    bst = xgb.cv(param, dX_base, num_round, nfold=5, metrics = {'auc'})    
    print 'It takes time = ', datetime.datetime.now() - start

else:
    bst = xgb.train(plst, dX_base, num_round)

    data = pd.read_csv('feature/outdate/all_total_test.csv', header=False)

    dX_test = xgb.DMatrix(data.drop('enrollment_id', axis=1).as_matrix())
    predicted = bst.predict(dX_test)
    results = data[['enrollment_id']].astype(int)
    results['predicted'] = pd.DataFrame(predicted)

    print results.shape, results.columns
    results.to_csv('results/results_0707_v3.csv', index=False, header=False)


