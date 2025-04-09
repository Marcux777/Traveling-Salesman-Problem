"""
Módulo que define a classe Individual que representa um indivíduo na população do algoritmo genético.
"""

import random
import numpy as np
from typing import List, Tuple, Optional, Set


class Individual:
    """
    Classe que representa um indivíduo no algoritmo genético para o Problema do Caixeiro Viajante.
    """

    def __init__(self, gene: List[int], distance_matrix: np.ndarray, tabu_list: Optional[List[Tuple[int, int]]] = None):
        """
        Inicializa um indivíduo com um gene (rota) e matriz de distâncias.

        Args:
            gene: Lista de cidades representando a rota
            distance_matrix: Matriz de distâncias entre as cidades
            tabu_list: Lista de movimentos proibidos
        """
        self.gene = np.array(gene, dtype=np.int32)
        self.distance_matrix = np.array(distance_matrix, dtype=np.float64)
        self.fitness = self.calculate_fitness()
        self.tabu_list = [] if tabu_list is None else tabu_list

    @staticmethod
    def create_random_path(n: int) -> List[int]:
        """
        Cria um caminho aleatório de comprimento n.

        Args:
            n: Número de cidades

        Returns:
            Lista com uma permutação aleatória dos números de 1 a n
        """
        path = list(range(1, n + 1))
        random.shuffle(path)
        return path

    def calculate_fitness(self) -> float:
        """
        Calcula o valor de fitness (distância total da rota) do indivíduo.

        Returns:
            Distância total da rota
        """
        gene_array = self.gene - 1  # Converter para indexação baseada em 0
        edges = np.column_stack((gene_array, np.roll(gene_array, -1)))
        return np.sum(self.distance_matrix[edges[:, 0], edges[:, 1]])

    def crossover(self, other: 'Individual') -> 'Individual':
        """
        Realiza crossover entre este indivíduo e outro para criar um novo indivíduo.
        Implementa o Ordered Crossover (OX).

        Args:
            other: Outro indivíduo para cruzamento

        Returns:
            Novo indivíduo resultante do crossover
        """
        child = self.gene.copy()
        cut_point1, cut_point2 = sorted(
            random.sample(range(1, len(self.gene)), 2))
        middle_parent2 = set(other.gene[cut_point1:cut_point2])

        child_pos = cut_point2
        for gene in other.gene:
            if gene not in middle_parent2:
                while child_pos < len(child) and child[child_pos] in middle_parent2:
                    child_pos += 1
                    if child_pos == len(child):
                        child_pos = 0
                if child_pos < len(child):
                    child[child_pos] = gene
                    child_pos += 1
                    if child_pos == len(child):
                        child_pos = 0

        return Individual(child, self.distance_matrix, self.tabu_list.copy())

    def multipoint_crossover(self, other: 'Individual', points: int = 2) -> 'Individual':
        """
        Realiza crossover de múltiplos pontos entre este indivíduo e outro.

        Args:
            other: Outro indivíduo para cruzamento
            points: Número de pontos de corte

        Returns:
            Novo indivíduo resultante do crossover
        """
        indices = sorted(random.sample(range(len(self.gene)), points))
        child = self.gene.copy()

        for i in range(len(indices) - 1):
            if i % 2 == 1:
                child[indices[i]:indices[i+1]
                      ] = other.gene[indices[i]:indices[i+1]]

        return Individual(child, self.distance_matrix, self.tabu_list.copy())

    def two_opt_gain(self, i: int, j: int) -> float:
        """
        Calcula o ganho de aplicar 2-opt entre as cidades de índices i e j.

        Args:
            i: Índice da primeira cidade
            j: Índice da segunda cidade

        Returns:
            Ganho em termos de distância (negativo significa melhoria)
        """
        a, b = self.gene[i] - 1, self.gene[i + 1] - 1
        c, d = self.gene[j] - 1, self.gene[(j + 1) % len(self.gene)] - 1

        current = self.distance_matrix[a, b] + self.distance_matrix[c, d]
        new = self.distance_matrix[a, c] + self.distance_matrix[b, d]

        return new - current

    def apply_two_opt(self) -> bool:
        """
        Aplica a heurística 2-opt para melhorar o indivíduo.

        Returns:
            Booleano indicando se houve melhoria
        """
        improved = False

        for i in range(len(self.gene) - 2):
            for j in range(i + 2, len(self.gene) - 1):
                gain = self.two_opt_gain(i, j)
                if gain < 0:
                    # Reverse the segment between i+1 and j
                    self.gene[i+1:j+1] = self.gene[i+1:j+1][::-1]
                    self.fitness += gain
                    improved = True

        return improved

    def mutate(self, swap_rate: float = 0.01, two_opt_rate: float = 0.1) -> None:
        """
        Aplica mutação ao indivíduo com uma taxa especificada.

        Args:
            swap_rate: Taxa de aplicação de mutação swap
            two_opt_rate: Taxa de aplicação da melhoria 2-opt
        """
        if random.random() < swap_rate:
            # Seleciona k trocas aleatórias para aplicar
            k = max(1, int(len(self.gene) * 0.005))
            swaps = random.sample(range(len(self.gene)), k * 2)

            for i in range(0, k * 2, 2):
                idx1, idx2 = swaps[i], swaps[i + 1]
                if abs(idx1 - idx2) > 1 and (idx1, idx2) not in self.tabu_list:
                    self.gene[idx1], self.gene[idx2] = self.gene[idx2], self.gene[idx1]
                    self.tabu_list.append((idx1, idx2))

                    # Limite o tamanho da lista tabu
                    if len(self.tabu_list) > 1000:
                        self.tabu_list.pop(0)

            # Recalcula o fitness apenas se alguma troca foi feita
            if k > 0:
                self.fitness = self.calculate_fitness()

        # Aplica 2-opt com probabilidade two_opt_rate
        if random.random() < two_opt_rate:
            self.apply_two_opt()

    def inversion_mutation(self, rate: float = 0.01) -> None:
        """
        Aplica mutação por inversão com uma taxa especificada.

        Args:
            rate: Taxa de aplicação da mutação
        """
        if random.random() < rate:
            i, j = sorted(random.sample(range(len(self.gene)), 2))
            self.gene[i:j+1] = self.gene[i:j+1][::-1]
            self.fitness = self.calculate_fitness()

        self.apply_two_opt()
