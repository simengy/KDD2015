import log_transform

enrollName = '../data/test/enrollment_test.csv'
logName = '../data/test/log_test.csv'
outFile = 'feature/log_feature_test_3.csv'

log_transform.read(enrollName, logName, outFile, 1000)
