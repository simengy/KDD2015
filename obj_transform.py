import pandas as pd
import numpy as np
import networkx as nx
import datetime
from constant import START_DATE

def depth(source, traversal, d=0):

    if source in traversal:
        
        for key in traversal[source]: 
            temp = depth(key, traversal, d+1)
            
            if d < temp:
                    d = temp

    return d

def members(source, traversal, count=0):

    if source in traversal:
        for key in traversal[source]:
            count += 1
            count += members(key, traversal, count)

    return count

def obj_transform(dataframe=None, G=None):

    dataframe1 = pd.concat([dataframe['enrollment_id'], pd.get_dummies(dataframe['category'])], axis=1)
    
    if G:
        betweenness = nx.betweenness_centrality(G)
        in_degree = nx.in_degree_centrality(G)
        out_degree = nx.out_degree_centrality(G)
        pagerank = nx.pagerank(G)
        pre = nx.bfs_predecessors(G, dataframe['module_id'][0])
        suc = nx.bfs_successors(G, dataframe['module_id'][0])
        
        nrow = dataframe.shape[0]

        graph_features = np.zeros((nrow, 8))
        
        for i in xrange(nrow):
            
            graph_features[i,0] = in_degree[dataframe['module_id'][i]]
            graph_features[i,1] = out_degree[dataframe['module_id'][i]] 
            graph_features[i,2] = betweenness[dataframe['module_id'][i]] 
            graph_features[i,3] = pagerank[dataframe['module_id'][i]]
            
            graph_features[i,4] = depth(dataframe['module_id'][i], suc)
            graph_features[i,5] = depth(dataframe['module_id'][i], pre)
            graph_features[i,6] = members(dataframe['module_id'][i], suc)
            graph_features[i,7] = members(dataframe['module_id'][i], pre)

        temp = pd.DataFrame(graph_features, index=dataframe.index)
        temp.columns = ['inDgree', 'outDegree', 'betweenness', 'pagerank',
                'depth', 'height', 'N_child', 'N_ancestor']
        temp['enrollment_id'] = dataframe['enrollment_id']
        temp.to_csv('checkpoint.csv')
        temp = temp.groupby('enrollment_id').aggregate(np.mean)
    # aggregating
    dataframe1 = dataframe1.groupby('enrollment_id').aggregate(np.sum)
    dataframe1 = pd.concat([dataframe1, temp], axis=1)
    
    return dataframe1

def graph(dataframe=None):

    G = nx.DiGraph()
    
    nrow = dataframe.shape[0]

    for i in xrange(nrow):
        
        source = dataframe['module_id'][i]
        G.add_node(source)
        
        if pd.isnull(dataframe['children'][i]) is not True:
            try:
                targets = dataframe['children'][i].split()
            except:
                raise ValueError('Data type is not correct:', i, 
                        dataframe.loc[i,], type(dataframe['children'][i]))
            
            for key in targets:
                G.add_edge(source, key)
    
    return G


def read(logName=None, objName=None, outFile=None, nrows=None):

    start = datetime.datetime.now()

    log_train = pd.read_csv(logName, header=False, nrows=nrows)
    obj_train = pd.read_csv(objName, header=False, nrows=nrows)
    
    G = graph(obj_train)
    # datetime to seconds
    log_train['time'] = (pd.to_datetime(log_train['time']) - START_DATE) / np.timedelta64(1, 's')
    #obj_train['start'] = (pd.to_datetime(obj_train['start']) - TIME_START) / np.timedelta64(1, 's')

    merged = pd.merge(log_train, obj_train[['module_id', 'category', 'children']], 
            left_on='object', right_on='module_id')

    print merged.columns, merged.shape

    obj_transform(merged, G).to_csv(outFile, index=True)

    print 'it takes time = ', datetime.datetime.now() - start

