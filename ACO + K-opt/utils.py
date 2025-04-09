"""
Módulo de funções utilitárias para o Problema do Caixeiro Viajante.
"""

import os
import numpy as np
import matplotlib.pyplot as plt


def calculate_distances(x, y):
    """
    Calcula a matriz de distâncias euclidiana entre todos os pares de cidades.

    Args:
        x (list): Lista de coordenadas x das cidades.
        y (list): Lista de coordenadas y das cidades.

    Returns:
        numpy.ndarray: Matriz de distâncias entre as cidades.
    """
    x = np.array(x)
    y = np.array(y)

    dx = x[:, np.newaxis] - x
    dy = y[:, np.newaxis] - y

    distances = np.sqrt(dx**2 + dy**2)

    np.fill_diagonal(distances, np.inf)  # Evita loops na mesma cidade

    return distances


def read_tsp_file(file_path):
    """
    Lê um arquivo TSP para obter as coordenadas das cidades.

    Args:
        file_path (str): Caminho para o arquivo TSP.

    Returns:
        tuple: Uma tupla contendo o número de cidades e as listas de coordenadas x e y.
    """
    x, y = [], []
    with open(file_path, "r") as file:
        n = int(next(file).strip())
        x = [float("inf")] * n
        y = [float("inf")] * n
        for line in file:
            l, c1, c2 = map(float, line.split())
            l = int(l) - 1
            x[l], y[l] = c1, c2
    return n, x, y


def find_tsp_file_path(filename, possible_folders=None):
    """
    Localiza o caminho completo para um arquivo TSP.

    Args:
        filename (str): Nome do arquivo TSP.
        possible_folders (list, optional): Lista de pastas possíveis para procurar o arquivo.

    Returns:
        str: Caminho completo para o arquivo TSP.
    """
    if possible_folders is None:
        possible_folders = ["../EntradasTSP/", "EntradasTSP/"]

    for folder in possible_folders:
        path = os.path.join(folder, filename)
        if os.path.exists(path):
            return path

    # Se não encontrar, retorna o primeiro caminho possível
    return os.path.join(possible_folders[0], filename)


def get_available_tsp_files(folder_path):
    """
    Obtém uma lista de arquivos TSP disponíveis.

    Args:
        folder_path (str): Caminho para a pasta com arquivos TSP.

    Returns:
        list: Lista de nomes de arquivos TSP disponíveis.
    """
    try:
        files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
        return files
    except FileNotFoundError:
        return []


def plot_tsp_solution(solution, x, y, cost, title=None):
    """
    Plota a solução do Problema do Caixeiro Viajante.

    Args:
        solution (list): Lista representando a rota solução.
        x (list): Lista de coordenadas x das cidades.
        y (list): Lista de coordenadas y das cidades.
        cost (float): Custo da solução.
        title (str, optional): Título personalizado para o gráfico.
    """
    plt.figure(figsize=(10, 8))

    # Converter para arrays NumPy para indexação
    x_array = np.array(x)
    y_array = np.array(y)

    # Plotar as cidades
    plt.scatter(x_array, y_array, c='blue', s=50)

    # Plotar a rota
    for i in range(len(solution)):
        j = (i + 1) % len(solution)
        plt.plot([x_array[solution[i]], x_array[solution[j]]],
                 [y_array[solution[i]], y_array[solution[j]]], 'r-', alpha=0.6)

    # Adicionar rótulos
    for i, (x_pos, y_pos) in enumerate(zip(x_array, y_array)):
        plt.text(x_pos, y_pos, str(i+1), fontsize=8, ha='center', va='center',
                 bbox=dict(facecolor='white', alpha=0.7))

    # Definir título
    if title:
        plt.title(title)
    else:
        plt.title(
            f'Solução do Problema do Caixeiro Viajante - Custo: {cost:.2f}')

    plt.xlabel('Coordenada X')
    plt.ylabel('Coordenada Y')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
