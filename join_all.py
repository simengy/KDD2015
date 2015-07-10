import pandas as pd

def join_all(enrollName, logName, objName,  outFile, riskName=None):
    
    enroll = pd.read_csv(enrollName, header=False)
    log = pd.read_csv(logName, header=False)
    obj = pd.read_csv(objName, header=False)

    merged = pd.merge(enroll, log, on='enrollment_id', how='inner')
    merged = pd.merge(merged, obj, on='enrollment_id', how='left')
    
    if riskName:
        risk = pd.read_csv(riskName, header=False)
        merged = pd.merge(merged, risk, on='enrollment_id', how='left')
    merged.to_csv(outFile, index=False)


if __name__ == '__main__':

    # train data
    enrollName = 'feature/enrollment_feature_5.csv'
    logName = 'feature/log_feature_5.csv'
    objName = 'feature/obj_feature_5.csv'
    #riskName = 'feature/riskTable_feature_5.csv'
    outFile = 'feature/all_total_5.csv'
    join_all(enrollName, logName, objName, outFile)
    
    # test data    
    enrollName = 'feature/enrollment_feature_test_5.csv'
    logName = 'feature/log_feature_test_5.csv'
    objName = 'feature/obj_feature_test_5.csv'
    #riskName = 'feature/riskTable_feature_test_5.csv'
    outFile = 'feature/all_total_test_5.csv'
    join_all(enrollName, logName, objName, outFile)

