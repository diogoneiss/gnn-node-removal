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
# isso deveria funcionar..
# from model import debug 


"""
Ver como fazer isso com as arestas e coletar a estatística
"""

np.set_printoptions(precision=5, suppress=True)
sns.set(style="whitegrid")

#no array print limit
np.set_printoptions(threshold=50)

debug_prints = True
debug_histogram = False


def remove_nodes(g, total, strategy=2.2, seed=-1 ):
    total = 2
    if seed != -1:
        np.random.seed(seed)
    else:
        np.random.seed(10)
 
 
    # generate a list of randomly selected nodes
    nodes = g.nodes().numpy()
   
    size =  g.num_nodes()

    if size <= total:
        print("Não é possível remover mais nós do que existem")
        return
    
    
    if strategy == 1:
        # 1: Uniformemente, shuffle
        np.random.shuffle(nodes)
        #get removal  nodes
        nodes = nodes[0:total]
        if debug_prints:
            print(f"Removendo aleatoriamente, sem peso, {len(nodes)} nos" )
        
    # 2.1: Aleatoriamente, com peso ponderado por grau de entrada   
    elif strategy == 2 or strategy == 2.1:
        degreeArray = g.in_degrees().numpy()
        removal_probability = degreeArray/degreeArray.sum()
        
        #so pra visualizacao
        sorted_indexes = np.argsort(degreeArray)[::-1]
        debug_sorted_degrees =  np.array(degreeArray)[sorted_indexes]
        
        if debug_prints:
            print("Maiores graus no momento: ", debug_sorted_degrees.tolist()[0:50])
            print("Maiores probabilidades de remover: ", np.array(removal_probability[sorted_indexes[0:50]]))
       
        nodes = np.random.choice(nodes, total, p=removal_probability, replace=False)
        
     # 2.2: Aleatoriamente, com peso ponderado por grau de saida   
    elif strategy == 2.2:
        degreeArray = g.out_degrees().numpy()
        removal_probability = degreeArray/degreeArray.sum()
        
        #so pra visualizacao
        sorted_indexes = np.argsort(degreeArray)[::-1]
        debug_sorted_degrees =  np.array(degreeArray)[sorted_indexes]
        
        if debug_prints:
            print("Maiores graus no momento: ", debug_sorted_degrees.tolist()[0:50])
            print("Maiores probabilidades de remover: ", np.array(removal_probability[sorted_indexes[0:50]]))

       
        nodes = np.random.choice(nodes, total, p=removal_probability, replace=False)
        
    # 3: Remover do maior pro menor grau 
    # versao de in degrees
    elif strategy == 3 or strategy == 3.1:
        degreeArray = g.in_degrees().numpy()
        print("NEXT ITERATION_____________________________________")
        print('Mean of degrees: ', degreeArray.sum()/len(degreeArray))
        print("Size of degree array: ", len(degreeArray))
        

       
        
        print("___")
        #reverse sort by degree
        
        #sortIndexes = np.argsort(degreeArray)[::-1].copy()
        #1st step: sort by degree	
        #sortIndexes = np.argsort(degreeArray)[-total:].copy()
        sortIndexes = np.argsort(degreeArray)[::-1].copy()
        
        
        print("Sorted indexes: ", sortIndexes.tolist())
        
        #2nd step: get degree value info
        debug_sorted_degrees =  np.array(degreeArray)[sortIndexes]
        
        

        degreeDict = list(zip(sortIndexes, debug_sorted_degrees))[0:10]
        print("DegreeDict: ", degreeDict)
        
        hist = pd.DataFrame(debug_sorted_degrees)
        hist.columns = ['degree count']
        y = hist.groupby("degree count").size()
        x = y.index
        print("number of nodes to be removed in round: ", total)
        print(y)
        
        # create histogram with x and y
        """
        plt.bar(x, y)
        plt.xlabel("Degree")
        plt.ylabel("Number of nodes")
        plt.title("Histogram of node degrees")
        plt.show()
        """
  
        #removed nodes
        #for i in sortIndexes[0:total]:
        #    print(f"[{i}: degree {g.in_degrees(i)}] ", end="")
        #slice first total nodes
        #nodes = sortIndexes[:total].copy()
        #3rd step: add to nodes array to future removal
        nodes = sortIndexes[0:total].copy()
        print(nodes.tolist())
        if debug_prints: 
            #garantir os removidos
            nodes_removidos = g.in_degrees(torch.tensor(nodes)).numpy().tolist()
            maiorGrau = max(nodes_removidos)
            menorGrau = min(nodes_removidos)
        
            print("\nMaiores graus no momento e nodes: ")
            #print(sorted.tolist()[0:50], sep='\t')
            print(nodes_removidos[0:total], sep='\t')
          
            
            print(f"Maior grau removido: {maiorGrau}")
            print(f"Menor grau removido: {menorGrau}")
           # print("Em indices: ")
           # print(nodes.tolist()[0:50], sep='\t')
        
    # 3: Remover do maior pro menor grau 
    # versao de out_degrees
    elif strategy == 3.2:
        degreeArray = g.out_degrees().numpy()
       
        #reverse sort by degree
        sortIndexes = np.argsort(degreeArray)[::-1]
        debug_sorted_degrees =  np.array(degreeArray)[sortIndexes]
        
        #nao adianta usar valores dos graus, preciso dos indices anyway
        nodes = sortIndexes[0:total] 
        
        # necessario essa gambiarra para nao dar erro de contiguidade (stride)
        # ver um jeito mais elegante?
        nodes = nodes - np.zeros_like(nodes)
        
        if debug_prints:
            print("Maiores graus no momento: ")
            print(debug_sorted_degrees.tolist()[0:total], sep='\t')
            print("Em indices: ")
            print(nodes.tolist()[0:total], sep='\t')
        
    
    #Aleatorio com peso
    #nodes = np.random.choice
    
    """
    for _ in range(0, limit):
        # escolhe um inteiro menor que o numero de nós, sem repetição
        # fazer um shuffle de n itens
        while (True):
            i = secrets.randbelow(size)
            if i not in nodes:
                nodes.append(i)
                break
    """

   # plot the removed nodes
    #plt.scatter(nodes,nodes, marker='x',cmap='viridis' )
    #plt.colorbar()
   # nodes = np.array(nodes)
    try:
        g.remove_nodes(torch.tensor(nodes, dtype=torch.int64), store_ids=True)
        
        nData =  pd.DataFrame(g.ndata)

       
    except ValueError as e:
        print("Erro com o array de entrada")
        print(e)
        print("Formato dos nos: ")
        print(nodes)
        quit()
    
    

   
    if debug_histogram:
       # print(nodes)
      bin_size = 50
       
      ax = sns.histplot(data=nodes, kde=True, binwidth=bin_size, color='blue')
      ax.set(ylabel =f'Nos no bin de {bin_size} removidos', xlabel ='Indice de remocao')   
      plt.show()
    #por algum motivo list nao funciona, precisa ser tensor
    # esse store ids deveria armazenar o id dos removidos mas a documentação foi confusa quanto a 
    if debug_prints:
        print(f"Aplicada estrategia {strategy}  Novo total: {g.num_nodes()}")
    
    return g, nodes
    