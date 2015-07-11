import pandas as pd
import numpy as np
import networkx as nx
import datetime
from constant import START_DATE


def depth(source, traversal, d=0):
   
    maxDepth = d
    
    if source in traversal:
        for key in traversal[source]:
            
            temp = depth(key, traversal, d+1)
            
            if maxDepth < temp:
                maxDepth = temp

    return maxDepth


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

        nrow = dataframe.shape[0]

        graph_features = np.zeros((nrow, 7))
        
        for i in xrange(nrow):
            
            graph_features[i,0] = in_degree[dataframe['module_id'][i]] * 5000.0
            graph_features[i,1] = out_degree[dataframe['module_id'][i]] * 5000.0
            graph_features[i,2] = betweenness[dataframe['module_id'][i]] * 5000.0
            graph_features[i,3] = pagerank[dataframe['module_id'][i]] * 5000.0
            
            #pre = nx.bfs_predecessors(G, dataframe['module_id'][i])
            suc = nx.bfs_successors(G, dataframe['module_id'][i])
            graph_features[i,4] = depth(dataframe['module_id'][i], suc) 
            graph_features[i,5] = len( list(nx.ancestors(G, dataframe['module_id'][i]) ))
            graph_features[i,6] = len( list(nx.descendants(G, dataframe['module_id'][i]) ))

        temp = pd.DataFrame(graph_features, index=dataframe.index)
        temp.columns = ['inDgree', 'outDegree', 'betweenness', 'pagerank',
                'depth', 'N_ancestor', 'N_child']
        temp['enrollment_id'] = dataframe['enrollment_id']
        temp.to_csv('debugDir/checkpoint.csv')
    
    # aggregating
    dataframe1 = dataframe1.groupby('enrollment_id').aggregate(np.sum)
    
    temp1 = temp.groupby('enrollment_id').aggregate(np.mean)
    nameList = []
    colName = ['inDgree', 'outDegree', 'betweenness', 'pagerank',
                'depth', 'N_ancestor', 'N_child']
    for name in colName:
        nameList.append(name + '_mean')
    temp1.columns = nameList
    dataframe1 = pd.concat([dataframe1, temp1], axis=1)
    
    temp1 = temp.groupby('enrollment_id').aggregate(np.std) 
    nameList = []
    for name in colName:
        nameList.append(name + '_std')
    temp1.columns = nameList
    dataframe1 = pd.concat([dataframe1, temp1], axis=1)

    temp1 = temp.groupby('enrollment_id').aggregate(np.min)
    nameList = []
    for name in colName:
        nameList.append(name + '_min')
    temp1.columns = nameList
    dataframe1 = pd.concat([dataframe1, temp1], axis=1)
    
    temp1 = temp.groupby('enrollment_id').aggregate(np.max)
    nameList = []
    for name in colName:
        nameList.append(name + '_max')
    temp1.columns = nameList
    dataframe1 = pd.concat([dataframe1, temp1], axis=1)
    
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

    # Sanity check
    selfLoop = G.selfloop_edges()
    assert len(selfLoop) == 0, ValueError('self loop:', selfLoop)
    assert nx.is_directed_acyclic_graph(G), valueError('loop exists!')

    return G


def read(logName=None, objName=None, outFile=None, nrows=None):

    start = datetime.datetime.now()

    log_train = pd.read_csv(logName, header=False, nrows=nrows)
    obj_train = pd.read_csv(objName, header=False, nrows=nrows)
    print obj_train.shape 
    G = graph(obj_train)

    merged = pd.merge(log_train, obj_train[['module_id', 'category', 'children']], 
            left_on='object', right_on='module_id')

    print merged.columns, merged.shape

    obj_transform(merged, G).to_csv(outFile, index=True)

    print 'it takes time = ', datetime.datetime.now() - start

