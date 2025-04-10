"""
Módulo de funções para visualização de soluções do Problema do Caixeiro Viajante.
"""

import numpy as np
import matplotlib.pyplot as plt


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


def plot_convergence(costs, title=None):
    """
    Plota a curva de convergência do algoritmo.

    Args:
        costs (list): Lista de custos ao longo das iterações.
        title (str, optional): Título personalizado para o gráfico.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(costs, 'b-', linewidth=2)
    plt.xlabel('Iteração')
    plt.ylabel('Custo')

    if title:
        plt.title(title)
    else:
        plt.title('Curva de Convergência do Algoritmo')

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_multiple_runs(all_costs, labels=None, title=None):
    """
    Plota curvas de convergência de múltiplas execuções para comparação.

    Args:
        all_costs (list): Lista de listas de custos para cada execução.
        labels (list, optional): Lista de rótulos para cada execução.
        title (str, optional): Título personalizado para o gráfico.
    """
    plt.figure(figsize=(12, 7))

    for i, costs in enumerate(all_costs):
        label = labels[i] if labels and i < len(labels) else f'Execução {i+1}'
        plt.plot(costs, linewidth=2, label=label)

    plt.xlabel('Iteração')
    plt.ylabel('Custo')
    plt.legend()

    if title:
        plt.title(title)
    else:
        plt.title('Comparação de Múltiplas Execuções')

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
