import enrollment_transform

logName = '../data/train/log_train.csv'
outFile = 'feature/enrollment_feature_3.csv'

enrollment_transform.read(logName, outFile, 10000)
