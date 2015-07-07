import pandas as pd
import numpy as np
import datetime
import log_transform


TIME_START = datetime.datetime(2012,1,1)


start = datetime.datetime.now()

enroll_train = pd.read_csv('../data/train/enrollment_train.csv', header=False)
log_train = pd.read_csv('../data/train/log_train.csv', header=False)

log_train['time'] = (pd.to_datetime(log_train['time']) - TIME_START) / np.timedelta64(1, 's')
merged1 = pd.merge(enroll_train, log_train, on=['enrollment_id'])
print merged1.columns, merged1.shape

log_transform.azureml_main_count(merged1).to_csv('feature/log_feature.csv', index=False)

print 'it takes time = ', datetime.datetime.now() - start
