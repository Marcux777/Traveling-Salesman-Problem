"""
Módulo que define o otimizador genético para o Problema do Caixeiro Viajante.
"""

import logging
import random
import time
import multiprocessing as mp
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

from .individual import Individual


logger = logging.getLogger("GeneticOptimizer")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)


class GeneticOptimizer:
    """
    Classe que implementa o otimizador genético para o Problema do Caixeiro Viajante.
    """

    def __init__(self,
                 population_size: int = 100,
                 elite_size: float = 0.1,
                 mutation_rate: float = 0.01,
                 two_opt_rate: float = 0.1,
                 crossover_rate: float = 0.7,
                 use_parallel: bool = True,
                 max_processes: Optional[int] = None):
        """
        Inicializa o otimizador genético.

        Args:
            population_size: Tamanho da população
            elite_size: Proporção da população que é considerada elite
            mutation_rate: Taxa de mutação
            two_opt_rate: Taxa de aplicação da melhoria 2-opt
            crossover_rate: Taxa de aplicação do crossover
            use_parallel: Indica se deve usar processamento paralelo
            max_processes: Número máximo de processos para execução paralela
        """
        self.population_size = population_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.two_opt_rate = two_opt_rate
        self.crossover_rate = crossover_rate
        self.use_parallel = use_parallel
        self.max_processes = max_processes if max_processes else min(
            mp.cpu_count(), 8)
        self.best_fitnesses = []

    def initialize_population(self, n_cities: int, distance_matrix: np.ndarray) -> List[Individual]:
        """
        Inicializa a população com indivíduos aleatórios.

        Args:
            n_cities: Número de cidades
            distance_matrix: Matriz de distâncias entre as cidades

        Returns:
            Lista de indivíduos inicializados
        """
        return [Individual(Individual.create_random_path(n_cities), distance_matrix)
                for _ in range(self.population_size)]

    def select_elite(self, population: List[Individual]) -> List[Individual]:
        """
        Seleciona os melhores indivíduos da população.

        Args:
            population: População atual

        Returns:
            Lista com os melhores indivíduos
        """
        sorted_population = sorted(population, key=lambda x: x.fitness)
        elite_count = int(self.elite_size * len(population))
        return sorted_population[:elite_count]

    @staticmethod
    def calculate_fitness_parallel(individual: Individual) -> float:
        """
        Calcula o fitness de um indivíduo (para processamento paralelo).

        Args:
            individual: Indivíduo a ser avaliado

        Returns:
            Valor de fitness
        """
        return individual.calculate_fitness()

    def evolve(self, population: List[Individual], generations: int = 100) -> Tuple[List[Individual], int]:
        """
        Evolui a população de indivíduos por um número específico de gerações.

        Args:
            population: População inicial
            generations: Número de gerações

        Returns:
            Tupla contendo população final e número da última geração
        """
        if self.use_parallel and self.max_processes > 1:
            return self._evolve_parallel(population, generations)
        else:
            return self._evolve_sequential(population, generations)

    def _evolve_sequential(self, population: List[Individual], generations: int) -> Tuple[List[Individual], int]:
        """
        Evolui a população de forma sequencial.

        Args:
            population: População inicial
            generations: Número de gerações

        Returns:
            Tupla contendo população final e número da última geração
        """
        self.best_fitnesses = []

        for generation in range(generations):
            # Ordenar por fitness (menor é melhor)
            population.sort(key=lambda x: x.fitness)

            # Registrar o melhor indivíduo
            self.best_fitnesses.append(population[0].fitness)

            # Log a cada 10 gerações
            if generation % 10 == 0:
                logger.info("Geração: %d Melhor caminho: %s\n Custo: %d\n",
                            generation, population[0].gene, round(population[0].fitness))

            # Selecionar a elite
            elite = self.select_elite(population)

            # Criar nova população
            new_population = []

            # Adicionar elite diretamente à nova população
            new_population.extend(elite)

            # Completar a população com novos indivíduos
            while len(new_population) < self.population_size:
                parent1, parent2 = random.sample(elite, 2)

                if random.random() < self.crossover_rate:
                    child = parent1.crossover(parent2)

                    if random.random() < self.mutation_rate:
                        child.mutate(self.mutation_rate, self.two_opt_rate)

                    new_population.append(child)

            # Atualizar a população
            population = new_population

        # Ordenar a população final
        population.sort(key=lambda x: x.fitness)

        return population, generations

    def _evolve_parallel(self, population: List[Individual], generations: int) -> Tuple[List[Individual], int]:
        """
        Evolui a população usando processamento paralelo.

        Args:
            population: População inicial
            generations: Número de gerações

        Returns:
            Tupla contendo população final e número da última geração
        """
        self.best_fitnesses = []

        for generation in range(generations):
            # Ordenar por fitness (menor é melhor)
            population.sort(key=lambda x: x.fitness)

            # Registrar o melhor indivíduo
            self.best_fitnesses.append(population[0].fitness)

            # Log a cada 10 gerações
            if generation % 10 == 0:
                logger.info("Geração: %d Melhor caminho: %s\n Custo: %d\n",
                            generation, population[0].gene, round(population[0].fitness))

            # Selecionar a elite
            elite = self.select_elite(population)

            # Criar nova população com operadores genéticos
            new_individuals = []

            while len(new_individuals) < self.population_size - len(elite):
                parent1, parent2 = random.sample(elite, 2)

                if random.random() < self.crossover_rate:
                    child = parent1.crossover(parent2)

                    if random.random() < self.mutation_rate:
                        child.mutate(self.mutation_rate, self.two_opt_rate)

                    new_individuals.append(child)

            # Usar um tamanho de chunk adequado para balancear a carga
            chunk_size = max(1, len(new_individuals) //
                             (self.max_processes * 4))

            # Calcular fitness em paralelo
            with mp.Pool(self.max_processes) as pool:
                fitnesses = pool.map(
                    self.calculate_fitness_parallel,
                    new_individuals,
                    chunksize=chunk_size
                )

            # Atualizar fitness dos novos indivíduos
            for individual, fitness in zip(new_individuals, fitnesses):
                individual.fitness = fitness

            # Combinar elite com os novos indivíduos
            population = elite + new_individuals

        # Ordenar a população final
        population.sort(key=lambda x: x.fitness)

        return population, generations

    def get_best_individual(self) -> Optional[Individual]:
        """
        Retorna o melhor indivíduo da última evolução.

        Returns:
            O melhor indivíduo ou None se não houve evolução
        """
        if not self.best_fitnesses:
            return None

        return [ind for ind in sorted(self.population, key=lambda x: x.fitness)][0] if hasattr(self, 'population') else None
