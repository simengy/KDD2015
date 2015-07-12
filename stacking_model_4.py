import pandas as pd
import numpy as np
import datetime

from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import xgboost as xgb


EXPORT = False
XGB_BASE = False

start = datetime.datetime.now()

data1 = pd.read_csv('feature/all_total_10.csv', header=False)
data2 = pd.read_csv('../data/cleaned_train.csv', header=False)
data2 = pd.merge(data2, data1, on='enrollment_id')
label = pd.read_csv('../data/train/truth_train.csv', header=False)

data1 = pd.read_csv('feature/all_total_7.csv', header=False)
cols = [col for col in data1.columns if 'risk' not in col]
data1 = data1[cols]

# log transformation of count
#data.iloc[:,31:115] = pd.DataFrame(np.log(data.iloc[:,31:115].as_matrix() + 1.0), columns = data.iloc[:,31:115].columns)
#data = data.iloc[:,:-4]
data1 = data1.fillna(0)
data2 = data2.fillna(0)

assert data1.shape[0] == label.shape[0], 'Sizes of features (%d) and labels (%d) do not match!' % (data1.shape[0], label.shape[0])
#print data.drop('enrollment_id', axis=1).mean()
#print data.drop('enrollment_id', axis=1).max()


# train data split
X1_train, X1_test, X2_train, X2_test, Y_train, Y_test = train_test_split(
    data1,
    data2,
    label,
    test_size = 0.7
)

print "Build model:"

param = {'bst:max_depth':10, 'bst:min_child_weight':2, 'bst:eta':0.012, 'silent':1, 'objective':'binary:logistic', 'subsample':0.6, 'colsample_bytree':0.9, 'eval_metric': 'auc'}
param['nthread'] = 26
plst = param.items()
plst += [('eval_metric', 'auc')]
num_round = 200
    
dX_base = xgb.DMatrix(X1_train.drop('enrollment_id', axis=1).as_matrix(), label = Y_train['label'].as_matrix())

   
if XGB_BASE == True:
    bst = xgb.train(param, dX_base, num_round)
    dX_test = xgb.DMatrix(X1_test.drop('enrollment_id', axis=1).as_matrix(), label = Y_test['label'].as_matrix())
    addon = bst.predict(dX_test)
        
    print 'XGB:', addon.shape, X2_test.shape
    print addon 
    X2_test['addon1'] = pd.DataFrame(list(addon), index=X2_test.index)

if XGB_BASE == False:
    clf = RandomForestClassifier(n_estimators=400, max_depth=6, n_jobs=20, verbose=True)
    clf = clf.fit(X=X1_train.drop('enrollment_id', axis=1).as_matrix(), y=Y_train['label'].as_matrix())
    addon = clf.predict_proba(X1_test.drop('enrollment_id', axis=1).as_matrix())

    print 'Random Forest:', addon.shape, X2_test.shape
    print addon 
    X2_test['addon2'] = pd.DataFrame(list(addon[:,1]), index=X2_test.index)

    clf = LogisticRegression(C=0.01, tol=1e-7)
    clf = clf.fit(X=X1_train.drop('enrollment_id', axis=1).as_matrix(), y=Y_train['label'].as_matrix())
    addon = clf.predict_proba(X1_test.drop('enrollment_id', axis=1).as_matrix())

    print 'Logistic:', addon.shape, X2_test.shape
    print addon 
    X2_test['addon3'] = pd.DataFrame(list(addon[:,1]), index=X2_test.index)
    
    
# Layer 2
param = {'bst:max_depth':7, 'bst:min_child_weight':6, 'bst:eta':0.009, 'silent':1, 'objective':'binary:logistic', 'subsample':0.5, 'colsample_bytree':0.5, 'eval_metric': 'auc'}
param['nthread'] = 20
num_round = 900

dX_base = xgb.DMatrix(X2_test.drop('enrollment_id', axis=1).as_matrix(), label = Y_test['label'].as_matrix())
    

if EXPORT == False:
    bst = xgb.cv(param, dX_base, num_round, nfold=5, metrics={'auc'})
    
    print 'It takes time = ', datetime.datetime.now() - start
    
else:
    bst = xgb.train(plst, dX_base, num_round)

    data1 = pd.read_csv('feature/all_total_test_10.csv', header=False)
    data2 = pd.read_csv('../data/cleaned_test.csv', header=False)
    data2 = pd.merge(data2, data1, on='enrollment_id')
    # log transformation of count
    #data.iloc[:,31:115] = pd.DataFrame(np.log(data.iloc[:,31:115].as_matrix() + 1.0), columns = data.iloc[:,31:115].columns)
    data2 = data2.fillna(0)
    #data = data.iloc[:,:-4]
    
    dX_test = xgb.DMatrix(data2.drop('enrollment_id', axis=1).as_matrix())
    predicted = bst.predict(dX_test)
    results = data2[['enrollment_id']].astype(int)
    results['predicted'] = pd.DataFrame(predicted)

    print results.shape, results.columns
    results.to_csv('results/results_0712_stacking_v1.csv', index=False, header=False)
