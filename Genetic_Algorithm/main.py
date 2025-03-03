import random
import math
import matplotlib.pyplot as plt
import time
import os
from GA import GA


def calcular_preco(n, x, y):
    preco = [[float('inf')]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                preco[i][j] = float(
                    math.sqrt((x[j] - x[i])**2 + (y[j] - y[i])**2))
    return preco


pasta = "/workspaces/Heuristicas/Heuristicas/EntradasTSP"


lista_arquivos = [
    'Djibouti.txt', 'Qatar.txt', 'Argentina.txt', 'Burma.txt', 'China.txt',
    'Egypt.txt', 'Finland.txt', 'Greece.txt', 'Honduras.txt', 'Luxembourg.txt',
]

'''
0 - Djibouti.txt
1 - Qatar.txt
2 - Argentina.txt
3 - Burma.txt
4 - China.txt
5 - Egypt.txt
6 - Finland.txt
7 - Greece.txt
8 - Honduras.txt
9 - Luxembourg.txt
'''

inicio = time.time()
n, x, y = GA.read_file(pasta+lista_arquivos[0])
preco = calcular_preco(n, x, y)
populacao = [GA(GA.create_path(n), preco) for _ in range(100)]
resultados, geracao = GA.evolve(populacao)
print("Geracao: ", geracao, " Melhor caminho:",
      resultados[0].gene, "\n Custo: ", round(resultados[0].fit), "\n")
fim = time.time()

print("O tempo de execução foi: ", fim - inicio, " segundos\n")
