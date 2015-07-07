from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score, train_test_split

import sys
sys.path.append('/home/simengy/git/xgboost/wrapper/')
import xgboost as xgb


y_columns = [name for name in train_with_labels.columns if name.startswith('y')]

X_numerical_base, X_numerical_meta, X_sparse_base, X_sparse_meta, y_base, y_meta = train_test_split(
    X_numerical, 
    X_sparse, 
    train_with_labels[y_columns].values,
    test_size = 0.5
)

X_meta = [] 
X_test_meta = []

print "Build meta"


param = {'bst:max_depth':8, 'bst:eta':0.2, 'silent':0, 'objective':'binary:logistic' }
        param['nthread'] = 16
        plst = param.items()
        #plst += [('eval_metric', 'auc')] # Multiple evals can be handled in this way
        plst += [('eval_metric', 'logloss')]
        num_round = 100
        
        dX_base = xgb.DMatrix(X_numerical_base, label = y)
        bst = xgb.train(plst, dX_base, num_round)
        
        dX_num_meta = xgb.DMatrix(X_numerical_meta)
        X_meta.append(bst.predict(dX_num_meta))
        
        dX_test_num = xgb.DMatrix(X_test_numerical)
        X_test_meta.append(bst.predict(dX_test_num))
        
        print i, 'xgboost = ', datetime.now() - t1
        t2 = datetime.now()
        
        logit = LogisticRegression(C=0.01, tol=0.000001)
        logit.fit(X_sparse_base, y)
        X_meta.append(logit.predict_proba(X_sparse_meta))
        X_test_meta.append(logit.predict_proba(X_test_sparse))
        
        print i, 'logit = ', datetime.now() - t2
        
X_meta = np.column_stack(X_meta)
X_test_meta = np.column_stack(X_test_meta)
