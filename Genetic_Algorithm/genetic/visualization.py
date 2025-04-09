"""
Módulo para visualização dos resultados do algoritmo genético aplicado ao Problema do Caixeiro Viajante.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional, Tuple
import matplotlib.animation as animation

from .individual import Individual


def plot_convergence(best_fitnesses: List[float], title: str = 'Convergência do Algoritmo Genético'):
    """
    Plota o gráfico de convergência do algoritmo genético.

    Args:
        best_fitnesses: Lista com o fitness do melhor indivíduo a cada geração
        title: Título do gráfico
    """
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(best_fitnesses)), best_fitnesses)
    plt.title(title)
    plt.xlabel('Geração')
    plt.ylabel('Melhor Fitness (Distância)')
    plt.grid(True)
    plt.show()


def plot_route(individual: Individual,
               x_coords: np.ndarray,
               y_coords: np.ndarray,
               title: str = 'Melhor Rota Encontrada'):
    """
    Plota a rota do caixeiro viajante.

    Args:
        individual: Indivíduo (solução) a ser plotado
        x_coords: Coordenadas x das cidades
        y_coords: Coordenadas y das cidades
        title: Título do gráfico
    """
    plt.figure(figsize=(10, 8))

    # Ajusta o gene para indexação baseada em zero
    route = individual.gene - 1

    # Plota as cidades
    plt.scatter(x_coords, y_coords, s=100, c='blue', edgecolors='black')

    # Plota a rota
    for i in range(len(route)):
        j = (i + 1) % len(route)
        plt.plot([x_coords[route[i]], x_coords[route[j]]],
                 [y_coords[route[i]], y_coords[route[j]]], 'r-')

    # Adiciona texto com o custo total
    plt.text(0.05, 0.05, f'Custo Total: {round(individual.fitness, 2)}',
             transform=plt.gca().transAxes, fontsize=12)

    # Adiciona números às cidades
    for i in range(len(x_coords)):
        plt.text(x_coords[i], y_coords[i], str(
            i+1), fontsize=10, fontweight='bold')

    plt.title(title)
    plt.grid(True)
    plt.show()


def animate_evolution(generations: List[Individual],
                      x_coords: np.ndarray,
                      y_coords: np.ndarray,
                      interval: int = 200,
                      title: str = 'Evolução do Algoritmo Genético'):
    """
    Cria uma animação da evolução do algoritmo genético.

    Args:
        generations: Lista com o melhor indivíduo de cada geração
        x_coords: Coordenadas x das cidades
        y_coords: Coordenadas y das cidades
        interval: Intervalo entre frames em milissegundos
        title: Título da animação
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    def update(frame):
        ax.clear()
        individual = generations[frame]
        route = individual.gene - 1

        # Plota as cidades
        ax.scatter(x_coords, y_coords, s=100, c='blue', edgecolors='black')

        # Plota a rota
        for i in range(len(route)):
            j = (i + 1) % len(route)
            ax.plot([x_coords[route[i]], x_coords[route[j]]],
                    [y_coords[route[i]], y_coords[route[j]]], 'r-')

        # Adiciona texto com o custo total e geração atual
        ax.text(0.05, 0.05, f'Geração: {frame}\nCusto: {round(individual.fitness, 2)}',
                transform=ax.transAxes, fontsize=12)

        # Adiciona números às cidades
        for i in range(len(x_coords)):
            ax.text(x_coords[i], y_coords[i], str(
                i+1), fontsize=10, fontweight='bold')

        ax.set_title(title)
        ax.grid(True)

    ani = animation.FuncAnimation(
        fig, update, frames=len(generations), interval=interval)
    plt.tight_layout()
    plt.show()

    return ani
