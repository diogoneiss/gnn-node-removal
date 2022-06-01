import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

removal_percentage = 0.05

data = pd.read_csv("results.csv")

""" Formato dos dados:
  Tecnica  Precisão
0       1  0.782123
1       1  0.754717
2       1  0.690202
3       1  0.650685
0     2.1  0.742042
1     2.1  0.703883
2     2.1  0.660326
3     2.1  0.607477
0     3.1  0.705044
1     3.1  0.575980
2     3.1  0.448000
3     3.1  0.386260

"""

# get unique values in Tecnica column
tecnicas = data['Tecnica'].unique()

# get length of tecnicas[0]


titulo = f"Diferentes técnicas para remoção de nós"
plt.title(titulo)


y = data


for j in tecnicas :
    fracoes = len(data.query('Tecnica == @j'))
    x = np.arange(removal_percentage, removal_percentage*(fracoes+1), removal_percentage)
    plt.plot(x, y.query("Tecnica == @j")["Precisão"], label="Tecnica: " + str(j))


plt.ylim(bottom=0)
plt.ylim(top=0.8)

#name x and y axis
plt.xlabel("Porcentagem de nós removidos")
plt.ylabel("Precisão")
plt.legend()
plt.grid()

plt.show()





## mostrando com variacao de seed

dataSeed = pd.read_csv("dataSeed.csv")
print(dataSeed)


variacoes_de_seed = dataSeed['Seed'].unique()
titulo = f"Tecnica: {1} | Precisão média após {variacoes_de_seed} variacoes de seed"
plt.title(titulo)

for j in variacoes_de_seed :
    fracoes = len(dataSeed.query('Seed == @j'))
    x = np.arange(removal_percentage, removal_percentage*(fracoes+1), removal_percentage)
    plt.plot(x, dataSeed.query("Seed == @j")["Precisão"], label="Seed: " + str(j))



plt.ylim(bottom=0)
plt.ylim(top=0.8)

#name x and y axis
plt.xlabel("Porcentagem de nós removidos")
plt.ylabel("Precisão")
plt.legend()
plt.grid()

plt.show()

    
    
print(data.describe())
print(data)