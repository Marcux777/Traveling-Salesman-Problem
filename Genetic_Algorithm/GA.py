import random
import math
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import logging

logger = logging.getLogger("GeneticAlgorithm")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

'''
GA: Class to represent an individual in the GA.
'''


class GA:
    best_fitnesses = []

    def __init__(self, gene, price, tabu_list=[]):
        self.gene = np.array(gene, dtype=np.int32)
        self.price = np.array(price, dtype=np.float64)
        self.fit = self.calculate_fitness()
        self.tabu_list = [] if tabu_list is None else tabu_list

    '''
    create_path(n): Creates a random path of length n.
    '''
    @staticmethod
    def create_path(n):
        path = list(range(1, n + 1))
        random.shuffle(path)
        return path

    '''
    calculate_fitness(): Calculates the fitness value of the individual.
    '''

    def calculate_fitness(self):
        gene_array = self.gene - 1
        edges = np.column_stack((gene_array, np.roll(gene_array, -1)))
        return np.sum(self.price[edges[:, 0], edges[:, 1]])

    '''
    crossover(other): Crosses the current individual with another individual to create a new individual.
    '''

    def crossover(self, other: 'GA') -> 'GA':
        """
        Crosses the current individual with another individual to create a new individual.

        Args:
            other (GA): The other individual to cross with.

        Returns:
            GA: The new individual.
        """
        child = self.gene[:]
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
        return GA(child, self.price, self.tabu_list)

    def multipoint_crossover(self, other, points=2):
        indices = sorted(random.sample(range(len(self.gene)), points))
        child = self.gene[:]
        for i in range(len(indices) - 1):
            if i % 2 == 1:
                child[indices[i]:indices[i+1]
                      ] = other.gene[indices[i]:indices[i+1]]
        return GA(child, self.price, self.tabu_list)
    '''
    two_opt(): Applies the 2-opt heuristic to improve the individual.
    '''

    def two_opt(self):
        improved = True
        while improved:
            improved = False
            for i in range(len(self.gene) - 2):
                for j in range(i + 2, len(self.gene)-1):
                    gain = self.two_opt_gain(i, j)
                    if gain < 0:
                        self.gene[i+1:j+1] = self.gene[i+1:j+1][::-1]
                        self.fit -= gain
                        improved = True

    '''
    two_opt_gain(i, j): Calculates the gain of applying 2-opt between cities i and j.
    '''

    def two_opt_gain(self, i, j):
        a, b, c, d = self.gene[i], self.gene[i +
                                             1], self.gene[j], self.gene[(j+1) % len(self.gene)]
        current = self.price[a-1][b-1] + self.price[c-1][d-1]
        new = self.price[a-1][c-1] + self.price[b-1][d-1]
        return new - current

    '''
    mutation(rate=0.01): Applies mutation to the individual with a specified rate.
    '''

    def mutation(self, rate=0.01, two_opt_rate=0.1):
        if random.random() < rate:
            # Select k random swaps to apply
            k = int(len(self.gene)*0.005)
            swaps = random.sample(range(len(self.gene)), k*2)
            for i in range(0, k*2, 2):
                i, j = swaps[i], swaps[i+1]
                if abs(i - j) > 1 and (i, j) not in self.tabu_list:
                    self.gene[i], self.gene[j] = self.gene[j], self.gene[i]
                    self.tabu_list.append((i, j))
                    if len(self.tabu_list) > 1000:
                        self.tabu_list.pop(0)
            # Only recalculate fitness if any swaps were made
            if k > 0:
                self.fit = self.calculate_fitness()
        if (random.random() < two_opt_rate):
            self.two_opt()

    def inversion_mutation(self, rate=0.01):
        if random.random() < rate:
            i, j = sorted(random.sample(range(len(self.gene)), 2))
            self.gene[i:j+1] = self.gene[i:j+1][::-1]
            self.fit = self.calculate_fitness()
        self.two_opt()

    '''
    read_file(file_path): Reads the TSP data from a file.
    '''
    @staticmethod
    def read_file(file_path):
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

    @staticmethod
    def plot_convergence():
        plt.figure(figsize=(10, 5))
        list = GA.best_fitnesses
        plt.plot(range(len(list)), list)
        plt.title('Convergence of the Genetic Algorithm')
        plt.xlabel('Generation')
        plt.ylabel('Best Fitness')
        plt.grid(True)
        plt.show()

    @classmethod
    def calculate_fitness_parallel(cls, individual):
        total_dist = sum(individual.price[individual.gene[i] - 1]
                         [individual.gene[i + 1] - 1] for i in range(len(individual.gene) - 1))
        return total_dist

    '''
    evolve(population, generations=100): Evolves the population of individuals for a specified number of generations.
    '''
    @classmethod
    def evolve(cls, population, generations=100):
        for generation in range(generations):
            population.sort(key=lambda x: x.fit)

            if generation % 10 == 0:
                logger.info("Generation: %d Best path: %s\n Cost: %d\n",
                            generation, population[0].gene, round(population[0].fit))

            new = []

            best = population[:int(0.1 * len(population))]

            while len(new) < len(population) - len(best):
                parent1, parent2 = random.sample(best, 2)
                child = parent1.crossover(parent2)
                if random.random() >= 0.43:
                    child.mutation()
                new.append(child)

            population = best + new
            cls.best_fitnesses.append(population[0].fit)
        return population, generation

    @classmethod
    def evolve_parallel(cls, population, generations=100):
        # Determine o número ideal de processos
        num_processes = min(mp.cpu_count(), 8)

        for generation in range(generations):
            population.sort(key=lambda x: x.fit)

            if generation % 10 == 0:
                logger.info("Generation: %d Best path: %s\n Cost: %d\n",
                            generation, population[0].gene, round(population[0].fit))

            new = []
            best = population[:int(0.1 * len(population))]

            while len(new) < len(population) - len(best):
                parent1, parent2 = random.sample(best, 2)
                child = parent1.crossover(parent2)
                child.mutation()
                new.append(child)

            # Usar um tamanho de chunk adequado para balancear a carga
            chunk_size = max(1, len(new) // (num_processes * 4))
            with mp.Pool(num_processes) as pool:
                fitnesses = pool.map(
                    cls.calculate_fitness_parallel, new, chunksize=chunk_size)

            for individual, fitness in zip(new, fitnesses):
                individual.fit = fitness

            population = best + new
            cls.best_fitnesses.append(population[0].fit)

        return population, generation
