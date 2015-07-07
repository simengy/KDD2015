import pandas as pd

from sklearn.metrics import auc, roc_curve
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import KFold, train_test_split

import sys
sys.path.append('/home/simengy/git/xgboost/wrapper/')
import xgboost as xgb

import datetime

start = datetime.datetime.now()

data = pd.read_csv('feature/all_total.csv', header=False)
label = pd.read_csv('../data/train/truth_train.csv', header=False)

print data.shape, label.shape
print data.dropna().shape, label.dropna().shape, data.drop('enrollment_id', axis=1).mean()

data = data.fillna(data.mean())

kf = KFold(label.shape[0], n_folds = 5, shuffle=True)


param = {'bst:max_depth':7, 'bst:eta':0.010, 'silent':0, 'objective':'binary:logistic' }
param['nthread'] = 20
plst = param.items()
plst += [('eval_metric', 'auc')]
num_round = 20

for train_index, test_index in kf:

    
    dX_base = xgb.DMatrix(data.loc[train_index].drop('enrollment_id', axis=1).as_matrix(), label = label.loc[train_index]['label'].as_matrix())
    bst = xgb.train(param, dX_base, num_round)
        
    dX_test = xgb.DMatrix(data.loc[test_index].drop('enrollment_id', axis=1).as_matrix())
    predicted = bst.predict(dX_test)
    fpr, tpr, thresholds = roc_curve(label.loc[test_index]['label'].as_matrix(), predicted)
    print fpr, tpr
    print auc(fpr, tpr)

    
# Performance Matrix
#print predicted

print 'It takes time = ', datetime.datetime.now() - start
