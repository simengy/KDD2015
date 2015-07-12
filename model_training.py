import pandas as pd
import numpy as np
import datetime

from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score, train_test_split

import xgboost as xgb


EXPORT = False


start = datetime.datetime.now()
merged = pd.read_csv('../data/cleaned_train.csv', header=False)
data = pd.read_csv('feature/all_total_10.csv', header=False)
data = pd.merge(merged, data, on='enrollment_id')
label = pd.read_csv('../data/train/truth_train.csv', header=False)


#cols = [col for col in data.columns if 'Course' not in col]
#data = data[cols]
# log transformation of count
#data.iloc[:,31:115] = pd.DataFrame(np.log(data.iloc[:,31:115].as_matrix() + 1.0), columns = data.iloc[:,31:115].columns)
#data = data.iloc[:,:-4]
data = data.fillna(0)

assert data.shape[0] == label.shape[0], 'Sizes of features (%d) and labels (%d) do not match!' % (data.shape[0], label.shape[0])
#print data.drop('enrollment_id', axis=1).mean()
#print data.drop('enrollment_id', axis=1).max()


# train data split
X_train, X_test, Y_train, y_test = train_test_split(
    data, 
    label,
    test_size = 0.0
)

print "Build model:"

param = {'bst:max_depth':7, 'bst:min_child_weight':4, 'bst:eta':0.010, 'silent':1, 'objective':'binary:logistic', 'subsample':0.5, 'colsample_bytree':0.5, 'eval_metric': 'auc'}
param['nthread'] = 24
plst = param.items()
plst += [('eval_metric', 'auc')]
num_round = 1000
    
dX_base = xgb.DMatrix(X_train.drop('enrollment_id', axis=1).as_matrix(), label = Y_train['label'].as_matrix())


if EXPORT == False:
    bst = xgb.cv(param, dX_base, num_round, nfold=5, metrics = {'auc'})    
    print 'It takes time = ', datetime.datetime.now() - start

else:
    bst = xgb.train(plst, dX_base, num_round)

    merged = pd.read_csv('../data/cleaned_test.csv', header=False)
    data = pd.read_csv('feature/all_total_test_10.csv', header=False)
    data = pd.merge(merged, data, on='enrollment_id')

    # log transformation of count
    #data.iloc[:,31:115] = pd.DataFrame(np.log(data.iloc[:,31:115].as_matrix() + 1.0), columns = data.iloc[:,31:115].columns)
    data = data.fillna(0)
    #data = data.iloc[:,:-4]
    
    dX_test = xgb.DMatrix(data.drop('enrollment_id', axis=1).as_matrix())
    predicted = bst.predict(dX_test)
    results = data[['enrollment_id']].astype(int)
    results['predicted'] = pd.DataFrame(predicted)

    print results.shape, results.columns
    results.to_csv('results/results_0712_v1.csv', index=False, header=False)
