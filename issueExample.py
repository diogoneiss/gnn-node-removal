import math
import random
import secrets
import time
import numpy as np
import torch
import dgl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pdb
from dgl.data import CoraGraphDataset
import gnn

def remove_nodes(g, total):
   


    degreeArray = g.in_degrees().numpy()
    
    print('Mean of degrees: ', degreeArray.sum()/len(degreeArray))
    print("Size of degree array: ", len(degreeArray))
    
    print("__________")
    
    #sort indexes and reverse, to get greater degrees first
    sortIndexes = np.argsort(degreeArray)[::-1].copy()


    #print("Sorted indexes: ", sortIndexes.tolist())

    #2nd step: get degree value info
    debug_sorted_degrees =  np.array(degreeArray)[sortIndexes]


    # indexes and degrees of 10 to be removed nodes
    degreeDict = list(zip(sortIndexes, debug_sorted_degrees))[0:10]
    #print("DegreeDict: ", degreeDict)

    #take all degrees in graph to dataframe and group by degree
    hist = pd.DataFrame(debug_sorted_degrees)
    hist.columns = ['degrees in graph, grouped']
    y = hist.groupby("degrees in graph, grouped").size()
    print("number of nodes to be removed in round: ", total)
    print(y)

    #slice the desired number of nodes from sorted indexes
    nodes = sortIndexes[0:total].copy()
    #print(nodes.tolist())

    removedNodesSearchedInGraph = g.in_degrees(torch.tensor(nodes)).numpy().tolist()
    maiorGrau = max(removedNodesSearchedInGraph)
    menorGrau = min(removedNodesSearchedInGraph)

    print("\nSorted degree removals:  ")
    print(removedNodesSearchedInGraph[0:total], sep='\t')
    
    
    print(f"Largest degree removed: {maiorGrau}")
    print(f"Smallest degree removed: {menorGrau}")
  

    g.remove_nodes(torch.tensor(nodes, dtype=torch.int64), store_ids=True)

    
    return g, nodes

dataset = CoraGraphDataset()[0]
precision = []
trainingEpochs = 60
nodeRemovalsPerRound = 50

for i in range(7):
    
    print(f"\n______________ITERATION #{i}______________________")
    g, removedNodes = remove_nodes(dataset, nodeRemovalsPerRound)
    currentPrecision = gnn.train(dataset, trainingEpochs)
    precision.append(currentPrecision)
    
for i in range(len(precision)):
    print(f"Precision of iteration {i+1}: {precision[i]}")
    