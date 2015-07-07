import pandas as pd

def join_all(enrollName, logName, objName, outFile):
    
    enroll = pd.read_csv(enrollName, header=False)
    log = pd.read_csv(logName, header=False)
    obj = pd.read_csv(objName, header=False)

    merged = pd.merge(enroll, log, on='enrollment_id', how='inner')
    pd.merge(merged, obj, on='enrollment_id', how='left').to_csv(outFile, index=False)


if __name__ == '__main__':

    # train data
    enrollName = 'feature/enrollment_feature_3.csv'
    logName = 'feature/log_feature_3.csv'
    objName = 'feature/obj_feature_3.csv'
    outFile = 'feature/all_total_3.csv'
    join_all(enrollName, logName, objName, outFile)
    
    # test data    
    enrollName = 'feature/enrollment_feature_test_3.csv'
    logName = 'feature/log_feature_test_3.csv'
    objName = 'feature/obj_feature_test_3.csv'
    outFile = 'feature/all_total_test_3.csv'
    join_all(enrollName, logName, objName, outFile)

