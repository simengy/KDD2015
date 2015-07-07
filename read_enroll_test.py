import pandas as pd
import numpy as np
import datetime
import enrollment_transform


TIME_START = datetime.datetime(2012,1,1)


start = datetime.datetime.now()
log_train = pd.read_csv('../data/test/log_test.csv', header=False)


log_train['time'] = (pd.to_datetime(log_train['time']) - TIME_START) / np.timedelta64(1, 's')
print log_train.columns, log_train.shape

enrollment_transform.azureml_main_enrollment(log_train).to_csv('feature/enrollment_feature_test.csv', index=False)

print 'it takes time = ', datetime.datetime.now() - start
