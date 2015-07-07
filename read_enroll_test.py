import enrollment_transform

logName = '../data/test/log_test.csv'
outFile = 'feature/enrollment_feature_test_3.csv'

enrollment_transform.read(logName, outFile, nrows=None)
