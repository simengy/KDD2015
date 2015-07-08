import pandas as pd
import numpy as np

from sklearn.preprocessing import normalize
from sklearn.metrics import auc, roc_curve
from sklearn.cross_validation import KFold, train_test_split

import sys
sys.path.append('/home/simengy/git/xgboost/wrapper/')
import xgboost as xgb

import datetime

start = datetime.datetime.now()

data = pd.read_csv('feature/all_total_5.csv', header=False)
label = pd.read_csv('../data/train/truth_train.csv', header=False)

# missing value
#data = data.fillna(0)
# sub-columns
cols = [col for col in data.columns if 'Course' not in col]
#data = data[cols]
#data = data.iloc[:,:-4]
# log transformation of count
#data.iloc[:,31:115] = pd.DataFrame(np.log(data.iloc[:,31:115].as_matrix() + 1.0), columns = data.iloc[:, 31:115].columns)

print data.shape, label.shape
print data.columns
#print data.iloc[:,31:115].columns
print data.dropna().shape, label.dropna().shape, 
# understand the metadata
stats=pd.DataFrame(data.drop('enrollment_id', axis=1).mean(), columns=['mean'])
stats['median'] = data.drop('enrollment_id', axis=1).median()
stats['min'] = data.drop('enrollment_id', axis=1).min()
stats['max'] = data.drop('enrollment_id', axis=1).max()
stats['std'] = data.drop('enrollment_id', axis=1).std()
stats.to_csv('mis/stats_train_5.csv')


kf = KFold(label.shape[0], n_folds = 5, shuffle=True)


param = {'bst:max_depth':8, 'bst:min_child_weight':1, 'bst:eta':0.008, 'silent':1, 'objective':'binary:logistic', 'subsample':0.5}
param['nthread'] = 30
plst = param.items()
plst += [('eval_metric', 'auc')]
num_round = 10
aucScore = []

for train_index, test_index in kf:
    
    temp_train = data.loc[train_index].drop('enrollment_id', axis=1).as_matrix()
    dX_base = xgb.DMatrix(temp_train, label = label.loc[train_index]['label'].as_matrix())
    temp_test = data.loc[test_index].drop('enrollment_id', axis=1).as_matrix()
    dX_test = xgb.DMatrix(temp_test)

    bst = xgb.train(param, dX_base, num_round)
    predicted = bst.predict(dX_test)
    fpr, tpr, thresholds = roc_curve(label.loc[test_index]['label'].as_matrix(), predicted)
    aucScore.append(auc(fpr, tpr))

# Summary
print np.mean(aucScore), '+-', np.std(aucScore)
print 'It takes time = ', datetime.datetime.now() - start
