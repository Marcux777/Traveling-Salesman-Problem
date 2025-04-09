"""
Pacote de algoritmo genético para resolver o Problema do Caixeiro Viajante.
"""

from .individual import Individual
from .optimizer import GeneticOptimizer
from .utils import read_file, calculate_distance_matrix
from .visualization import plot_convergence, plot_route, animate_evolution

__all__ = ['Individual', 'GeneticOptimizer', 'read_file', 'calculate_distance_matrix',
           'plot_convergence', 'plot_route', 'animate_evolution']
