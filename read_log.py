import log_transform

enrollName = '../data/train/enrollment_train.csv'
logName = '../data/train/log_train.csv'
outFile = 'feature/log_feature_trial.csv'

log_transform.read(enrollName, logName, outFile, nrows=10000)
