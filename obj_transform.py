import pandas as pd
import numpy as np

def transform(dataframe=None):

    dataframe1 = pd.concat([dataframe['enrollment_id'], pd.get_dummies(dataframe['category'])], axis=1)
    
    return  dataframe1.groupby('enrollment_id').aggregate(np.sum)
