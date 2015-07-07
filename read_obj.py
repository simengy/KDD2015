import obj_transform

logName = '../data/train/log_train.csv'
objName = '../data/object.csv'
outName = 'feature/obj_feature_3.csv'

obj_transform.read(logName, objName, outName, nrows=None)
