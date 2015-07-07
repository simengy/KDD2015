import pandas as pd

enroll = pd.read_csv('feature/enrollment_feature_test.csv', header=False)
log = pd.read_csv('feature/log_feature_test.csv', header=False)
obj = pd.read_csv('feature/obj_feature_test.csv', header=False)

merged = pd.merge(enroll, log, on='enrollment_id', how='inner')
pd.merge(merged, obj, on='enrollment_id', how='left').to_csv('feature/all_total_test.csv', index=False)
