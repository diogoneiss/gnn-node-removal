# call modifyData.py to modify the data

import sys
from tabnanny import verbose
import modifyDataset
import gnn
import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.data import CoraGraphDataset
from dgl.nn import GraphConv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset = CoraGraphDataset()[0]
precision = []

#hyperparameters
quantidade_remocoes_de_nos = 19
total_epochs = 60 # mais ou menos depois de 40 estabiliza
removal_percentage = 0.05
variacoes_de_seed = 10
# end hyperparameters

test = True

#debug global variable
global debug 


def modelDriver():
    node_removal_options = [1, 2.1, 3.1]
    
    if test:
        global quantidade_remocoes_de_nos
        quantidade_remocoes_de_nos = 8
        node_removal_options = [3]
   
    
    nodesRemoved = []
   
    data = pd.DataFrame(columns=['Tecnica', 'Precisão'])
   
    nodes_per_iteration = int(dataset.num_nodes() * removal_percentage)
    # variacao de tecnicas
    for i in node_removal_options:
        
        g = CoraGraphDataset(verbose=False)[0]
        precision, nodeRemovals = trainModelInstance(g, nodes_per_iteration, i)
        
        #insert precision array and  expanded i into data
        #create array with len(precision) and fill with i
        tecnica = np.full(len(precision), i)
        data = data.append(pd.DataFrame({'Tecnica': tecnica, 'Precisão': precision}))
        nodesRemoved.append(nodeRemovals)
        
    ## Testar a heuristica aleatoria    
    seedPrecisions = []
    dataSeed = pd.DataFrame(columns=['Seed', 'Precisão'])
    if False:
        for i in range(variacoes_de_seed):
        
            g = CoraGraphDataset(verbose=False)[0]
            tmpPrecision = randomTraining(g, quantidade_remocoes_de_nos, i)
            seedPrecisions.append(tmpPrecision)
            # append no dataSeed
            seed = np.full(len(tmpPrecision), i)
            dataSeed = dataSeed.append(pd.DataFrame({'Seed': seed, 'Precisão': tmpPrecision}))
            
            
            
        dataSeed.to_csv("dataSeed.csv")
        
    
    
    #save csv of data
    #data.to_csv('results.csv', index=False)
    
    
    #nodeDf = pd.DataFrame(np.array(nodeRemovals).T)
   
    """
    g = sns.FacetGrid(nodeDf)
    g.map(sns.histplot,  kde=True, bins=10, color='blue')
    plt.show()
   """

def trainModelInstance(g, nodes_per_iteration=100, nodeRemovalStrategy=1):
    ## TODO: Começar sem remocao, pra melhores resultados
    nodeRemovals = []
    precision = []
    contador = 0
    print("------------------------------------------------")
    print(f"Testando com a tecnica {nodeRemovalStrategy} e {nodes_per_iteration} remocoes por iteracao")
    
    for _ in range(quantidade_remocoes_de_nos):
        contador += 1
        
        g, removedNodes = modifyDataset.remove_nodes(g, nodes_per_iteration, nodeRemovalStrategy)
        currentPrecision = gnn.train(g, total_epochs)
        #currentPrecision = (i+1) / ((i+1) * (i+1))
        precision.append(currentPrecision)
        nodeRemovals.append(removedNodes)
        
        
           
    titulo = f"Tecnica: {nodeRemovalStrategy} | Precisão após {contador} remoções, totalizando {nodes_per_iteration*contador} nós retirados"
    plt.title(titulo)
    
    #create vector of increments of nodes_per_iteration
    
    
    
    x = np.arange(0, removal_percentage*contador, removal_percentage)
    y = precision
    plt.plot(x, y, "bo-")
    plt.ylim(bottom=0)
    plt.ylim(top=1)
    
    #name x and y axis
    plt.xlabel("Porcentagem de nós removidos")
    plt.ylabel("Precisão")
    
    # adicionar pontos descritivos
    for x1,y1 in zip(x,y):

        label = "{:.2f}".format(y1)

        plt.annotate(label, 
                    (x1,y1),
                    textcoords="offset points",
                    xytext=(0,10), 
                    ha='center') 
    

    plt.show()
    return precision, nodeRemovals

def randomTraining(g, nodes_per_iteration=100, seed=10):
    nodeRemovals = []
    precision = []
    contador = 0
    
    print("------------------------------------------------")
    print(f"Testando variacao de seeds com a tecnica {1} e seed {seed}")
    for i in range(quantidade_remocoes_de_nos):
        contador += 1
        # Necessario "resetar" o grafo
        
        data, removedNodes = modifyDataset.remove_nodes(g, nodes_per_iteration, 1, seed)
        currentPrecision = gnn.train(data, total_epochs)
        #currentPrecision = (i+1) / ((i+1) * (i+1))
        precision.append(currentPrecision)
        nodeRemovals.append(removedNodes)
        
        
    return precision

if __name__ == "__main__":
 
    # if -d is passed, debug is true
    if "-d" in sys.argv:
        debug = True
    else:
        debug = False
    
    
    modelDriver()
