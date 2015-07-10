from numpy import *
from datetime import datetime
import pytz
import pandas as pd
from constant import START_DATE


def dumpRiskTable(dataframe1 = None, dataframe2 = None):
    
    numrows = dataframe1.shape[0]
    
    subset = dataframe1[['label', 'course_id']]
    course_risk = subset.groupby('course_id').aggregate(mean)
    
    dataframe1['weekday'] = dataframe1['time'].apply(lambda x: x.weekday())
    dataframe1['hour'] = dataframe1['time'].apply(lambda x: x.hour)
    subset = dataframe1[['label', 'weekday', 'hour']]
    time_risk = subset.groupby(['weekday', 'hour']).aggregate(mean)
    
    course_risk.to_csv('risktable/course_risk.csv', index=True)
    time_risk.to_csv('risktable/time_risk.csv', index=True)


def lookup(enrollName = None, logName = None, outFile = None, nrows = None):

    start = datetime.now()
    
    dataframe = pd.read_csv(enrollName, header=False, nrows=nrows)
    dataframe1 = pd.read_csv(logName, header=False, nrows=nrows)
    dataframe = pd.merge(dataframe, dataframe1, on=['enrollment_id'])
    
    course_risk = pd.read_csv('risktable/course_risk.csv', header=False)
    time_risk = pd.read_csv('risktable/time_risk.csv', header=False)
    course_risk = course_risk.set_index('course_id').to_dict()['label']
    time_risk = time_risk.set_index(['weekday','hour']).to_dict()['label']
    
    dataframe['time'] = pd.to_datetime(dataframe['time'])
    dataframe['weekday'] = dataframe['time'].apply(lambda x: x.weekday())
    dataframe['hour'] = dataframe['time'].apply(lambda x: x.hour)
    
    numrows = dataframe1.shape[0]
    feature_matrix = zeros((numrows, 3))
    
    for i in range(numrows):
        feature_matrix[i,0] = dataframe['enrollment_id'][i]
        feature_matrix[i,1] = course_risk[dataframe['course_id'][i]]
        feature_matrix[i,2] = time_risk[dataframe['weekday'][i], dataframe['hour'][i]]
    print 'I am done'
    cols = ['enrollment_id', 'time_risk', 'course_risk']
    dataframe = pd.DataFrame(feature_matrix, columns=cols)
    temp = dataframe[cols].groupby('enrollment_id').aggregate(mean)
    temp = pd.concat((temp, dataframe[cols].groupby('enrollment_id').aggregate(std)), axis=1)
    temp.columns = ['time_risk_mean',
     'course_risk_mean',
     'time_risk_std',
     'course_risk_std']
    
    temp.to_csv(outFile, index=True)

    print 'it takes time = ', datetime.now() - start


def read(enrollName = None, logName = None, label = None, nrows = None):

    start = datetime.now()
    enroll_train = pd.read_csv(enrollName, header=False, nrows=None)
    log_train = pd.read_csv(logName, header=False, nrows=nrows)
    label_train = pd.read_csv(label, header=False, nrows=None)
    log_train['time'] = pd.to_datetime(log_train['time'])
    merged = pd.merge(label_train, enroll_train, on=['enrollment_id'])
    merged = pd.merge(merged, log_train, on=['enrollment_id'])
    print merged.columns, merged.shape
    dumpRiskTable(merged)
    print 'it takes time = ', datetime.now() - start
