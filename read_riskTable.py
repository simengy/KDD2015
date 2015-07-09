import riskTable

logName = '../data/train/log_train.csv'
enrollName = '../data/train/enrollment_train.csv'
labelName = '../data/train/truth_train.csv'
outName = 'feature/riskTable_feature_test.csv'

riskTable.read(enrollName, logName, labelName, outName, nrows=None)
