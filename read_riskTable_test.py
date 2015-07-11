import riskTable

logName = '../data/test/log_test.csv'
enrollName = '../data/test/enrollment_test.csv'
labelName = '../data/test/truth_test.csv'
outName = 'feature/riskTable_feature_test.csv'

riskTable.read(enrollName, logName, labelName, outName, nrows=None)
