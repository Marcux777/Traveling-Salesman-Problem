"""
Módulo com funções utilitárias para o algoritmo genético do Problema do Caixeiro Viajante.
"""

import numpy as np
from typing import Tuple, List


def read_file(file_path: str) -> Tuple[int, np.ndarray, np.ndarray]:
    """
    Lê os dados do TSP a partir de um arquivo.

    Args:
        file_path: Caminho do arquivo de entrada

    Returns:
        Tupla com número de cidades e coordenadas x e y
    """
    with open(file_path, 'r') as f:
        n = int(f.readline().strip())

    # Lê os dados a partir da segunda linha
    data = np.loadtxt(file_path, skiprows=1)

    # Cria vetores para as coordenadas
    x = np.empty(n, dtype=float)
    y = np.empty(n, dtype=float)

    # Preenche os arrays com base no índice (assumindo que a primeira coluna indica o índice da cidade)
    for row in data:
        idx, c1, c2 = row
        x[int(idx)-1] = c1
        y[int(idx)-1] = c2

    return n, x, y


def calculate_distance_matrix(n: int, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Calcula a matriz de distâncias entre todas as cidades.

    Args:
        n: Número de cidades
        x: Coordenadas x
        y: Coordenadas y

    Returns:
        Matriz de distâncias
    """
    coords = np.column_stack((x, y))

    # Calcula a norma (distância euclidiana) entre todos os pares de pontos
    dists = np.linalg.norm(
        coords[:, np.newaxis, :] - coords[np.newaxis, :, :], axis=2)

    # Define a diagonal como infinito para manter a lógica original
    np.fill_diagonal(dists, float('inf'))

    return dists
