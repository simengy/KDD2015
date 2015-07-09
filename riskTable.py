from numpy import *
from datetime import datetime
import pytz
import pandas as pd
from constant import START_DATE


def dumpRiskTable(dataframe1 = None, dataframe2 = None):
    
    unique_enrollment = list(set(dataframe1.iloc[:, 0]))
    num_unique_enrollment = len(unique_enrollment)
    enrollment_dict = dict(zip(unique_enrollment, range(num_unique_enrollment)))
    numrows = dataframe1.shape[0]
    count_features = zeros((num_unique_enrollment, 207))
    
    source_dict = {'browser': 0,
     'server': 1}
    event_dict = dict(zip(['access',
     'problem',
     'page_close',
     'nagivate',
     'video',
     'discussion',
     'wiki'], range(7)))
    course_dict = dict(zip(pd.Categorical(dataframe1['course_id']).unique(), range(39)))
    
    subset = dataframe1[['label', 'course_id']]
    course_risk = subset.groupby('course_id').aggregate(mean)
    
    dataframe1['weekday'] = dataframe1['time'].apply(lambda x: x.weekday())
    dataframe1['hour'] = dataframe1['time'].apply(lambda x: x.hour)
    
    subset = dataframe1[['label', 'weekday', 'hour']]
    time_risk = subset.groupby(['weekday', 'hour']).aggregate(mean)
    
    course_risk.to_csv('risktable/course_risk.csv', index=True)
    time_risk.to_csv('risktable/time_risk.csv', index=True)


def lookup(dataframe = None):
    numrows = dataframe.shape[0]
    course_risk = pd.read_csv('risktable/course_risk.csv', header=False)
    time_risk = pd.read_csv('risktable/time_risk.csv', header=False)
    dataframe['weekday'] = dataframe['time'].apply(lambda x: x.weekday())
    dataframe['hour'] = dataframe['time'].apply(lambda x: x.hour)
    for i in range(numrows):
        dataframe['course_risk'][i] = course_risk[dataframe['course_id'][i]]
        dataframe['time_risk'][i] = time_risk[dataframe['weekday'][i], dataframe1['hour'][i]]

    cols = ['enrollment_id', 'time_risk', 'course_risk']
    temp = dataframe[cols].groupby('enrollment_id').aggregate(mean)
    temp = pd.concat((temp, dataframe[cols].groupby('enrollment_id').aggregate(std)), axis=1)
    temp.columns = ['time_risk_mean',
     'course_risk_mean',
     'time_risk_std',
     'course_risk_std']
    return temp


def read(enrollName = None, logName = None, label = None, outFile = None, nrows = None):
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
