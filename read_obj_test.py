import obj_transform

logName = '../data/test/log_test.csv'
objName = '../data/object.csv'
outName = 'feature/obj_feature_test_3.csv'

obj_transform.read(logName, objName, outName, nrows=None)
