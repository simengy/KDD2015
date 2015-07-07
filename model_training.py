import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score, train_test_split

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

X_train, X_test, Y_train, y_test = train_test_split(
    data, 
    label,
    test_size = 0.0
)

X_meta = [] 
X_test_meta = []



print "Build meta"

param = {'bst:max_depth':15, 'bst:eta':0.012, 'silent':0, 'objective':'binary:logistic' }
param['nthread'] = 30
plst = param.items()
plst += [('eval_metric', 'auc')]
num_round = 800
        
dX_base = xgb.DMatrix(X_train.drop('enrollment_id', axis=1).as_matrix(), label = Y_train['label'].as_matrix())
#bst = xgb.train(plst, dX_base, num_round)
bst = xgb.cv(param, dX_base, num_round, nfold=5, metrics = {'auc'})
        
#dX_test = xgb.DMatrix(X_test.drop('enrollment_id', axis=1).as_matrix())
#predicted = bst.predict(dX_test)

# Performance Matrix
#print predicted

print 'It takes time = ', datetime.datetime.now() - start
