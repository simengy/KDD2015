import pandas as pd
import numpy as np
import datetime
import obj_transform


TIME_START = datetime.datetime(2012,1,1)


start = datetime.datetime.now()
obj_train = pd.read_csv('../data/object.csv', header=False)
log_train = pd.read_csv('../data/test/log_test.csv', header=False)

# datetime to seconds
log_train['time'] = (pd.to_datetime(log_train['time']) - TIME_START) / np.timedelta64(1, 's')
#obj_train['start'] = (pd.to_datetime(obj_train['start']) - TIME_START) / np.timedelta64(1, 's')

merged = pd.merge(log_train, obj_train[['module_id','category','children']], left_on='object', right_on='module_id')

print merged.columns, merged.shape

obj_transform.transform(merged).to_csv('feature/obj_feature_test_2.csv', index=True)

print 'it takes time = ', datetime.datetime.now() - start
