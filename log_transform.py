# The script MUST contain a function named azureml_main
# which is the entry point for this module.
#
# The entry point function can contain up to two input arguments:
#   Param<dataframe1>: a pandas.DataFrame
#   Param<dataframe2>: a pandas.DataFrame
# This module develop the counting features of log events.
# The output is a dataframe with 41 columns:
# Col1: enrollment id
# Cols 2-8: counts of events in Monday to Sunday
# Cols 9-32: counts of events in hour 0-23
# Cols 33-39: counts of event types
# Cols 40-41: counts of source types
# Cols 42-80: counts of course types
from numpy import *
from datetime import datetime
import pytz
import pandas as pd
from constant import START_DATE


def main_count(dataframe1 = None, dataframe2 = None):

    # Execution logic goes here
    unique_enrollment = list(set(dataframe1.iloc[:,0]))
    
    # Define dictionaries to map enrollment_id to row indices
    num_unique_enrollment = len(unique_enrollment)
    enrollment_dict = dict(zip(unique_enrollment,range(num_unique_enrollment)))
    numrows = dataframe1.shape[0]
    count_features = zeros((num_unique_enrollment, 80))
    
    # Define dictionaries to map source names, event types and course to indices
    source_dict = {'browser':0,'server':1}
    event_dict = dict(zip(['access','problem','page_close',\
        'nagivate','video','discussion','wiki'],range(7)))
    course_dict = dict(zip(pd.Categorical(dataframe1['course_id']).unique(),range(39)))
    
    for i in range(numrows):
        
        enrollment_id = dataframe1.iloc[i,0]
        row_index = enrollment_dict[enrollment_id]
        count_features[row_index,0] = enrollment_id
        timestamp_i = float(datafraime1.iloc[i,3])
        from_i = float(datafraime1.iloc[i,7])
        to_i = float(datafraime1.iloc[i,8])
        dateobj = datetime.fromtimestamp(timestamp_i, tz=pytz.utc)
        
        # time decay: 10 days no login is dropping out


        weekday = dateobj.weekday()
        hour = dateobj.hour
        #weekday is between 0 and 6, where Monday is 0, and Sunday is 6
        count_features[row_index,weekday+1] += 1  
        # hour is between 0 and 23
        count_features[row_index,hour+8] += 1
        
        event_index = event_dict[dataframe1.iloc[i,5]]
        source_index = source_dict[dataframe1.iloc[i,4]]
        count_features[row_index,event_index+32] += 1
        count_features[row_index,source_index+39] += 1
        
        # course is between 0 and 38
        course_index = course_dict[dataframe1.iloc[i,2]]
        count_features[row_index,course_index+41] += 1
        
    dataframe1 = pd.DataFrame(count_features)
    
    dataframe1.columns = ['enrollment_id','MonCount',\
        'TueCount','WedCount','ThuCount','FriCount',\
        'SatCount','SunCount','Hr0Count','Hr1Count','Hr2Count',\
        'Hr3Count','Hr4Count','Hr5Count','Hr6Count',\
        'Hr7Count','Hr8Count','Hr9Count','Hr10Count',\
        'Hr11Count','Hr12Count','Hr13Count','Hr14Count',\
        'Hr15Count','Hr16Count','Hr17Count','Hr18Count',\
        'Hr19Count','Hr20Count','Hr21Count','Hr22Count',\
        'Hr23Count','AccCount','ProCount','PagCount',\
        'NagCount','VidCount','DisCount','WikCount',\
        'BroCount','SerCount', \
        'Course0Count', 'Course1Count', 'Course2Count', 'Course3Count', 'Course4Count', 'Course5Count', \
        'Course6Count', 'Course7Count', 'Course8Count', 'Course9Count', 'Course10Count', 'Course11Count', \
        'Course12Count', 'Course13Count', 'Course14Count', 'Course15Count', 'Course16Count', 'Course17Count', \
        'Course18Count', 'Course19Count', 'Course20Count', 'Course21Count', 'Course22Count', 'Course23Count', \
        'Course24Count', 'Course25Count', 'Course26Count', 'Course27Count', 'Course28Count', 'Course29Count', \
        'Course30Count', 'Course31Count', 'Course32Count', 'Course33Count', 'Course34Count', 'Course35Count', \
        'Course36Count', 'Course37Count', 'Course38Count']    
    
    return dataframe1


def read(enrollName=None, logName=None, outFile=None, nrows=None): 
    
    start = datetime.now()
    
    enroll_train = pd.read_csv(enrollName, header=False, nrows=nrows)
    log_train = pd.read_csv(logName, header=False, nrows=nrows)
    course_date = pd.read_csv('../data/date.csv', header=False, nrows=nrows)
    
    log_train['time'] = (pd.to_datetime(log_train['time']) - START_DATE) / timedelta64(1, 's')
    course_date['from'] = (pd.to_datetime(course_date['from']) - START_DATE) / timedelta64(1, 's')
    course_date['to'] = (pd.to_datetime(course_date['to']) - START_DATE) / timedelta64(1, 's')
    merged = pd.merge(enroll_train, log_train, on=['enrollment_id'])
    merged = pd.merge(merged, course_date, on=['course_id'])
    print merged.columns, merged.shape

    main_count(merged).to_csv(outFile, index=False)

    print 'it takes time = ', datetime.now() - start


