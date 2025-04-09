"""
Módulo que implementa otimização bayesiana para os hiperparâmetros do algoritmo genético.
"""

import time
import numpy as np
from typing import Dict, Any, Callable, List, Tuple, Optional
import matplotlib.pyplot as plt
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.plots import plot_convergence, plot_objective

from genetic import Individual, GeneticOptimizer, read_file, calculate_distance_matrix


class BayesianOptimizer:
    """
    Classe que implementa otimização bayesiana para os hiperparâmetros do algoritmo genético.
    """

    def __init__(self,
                 input_file: str,
                 n_calls: int = 20,
                 n_random_starts: int = 5,
                 n_generations: int = 200,
                 verbose: bool = True):
        """
        Inicializa o otimizador bayesiano.

        Args:
            input_file: Caminho para o arquivo de entrada
            n_calls: Número de chamadas para a otimização bayesiana
            n_random_starts: Número de pontos aleatórios iniciais
            n_generations: Número de gerações para cada execução do algoritmo genético
            verbose: Se deve exibir informações de progresso
        """
        self.input_file = input_file
        self.n_calls = n_calls
        self.n_random_starts = n_random_starts
        self.n_generations = n_generations
        self.verbose = verbose
        self.best_params = None
        self.best_fitness = float('inf')
        self.results = None

        # Carrega os dados do arquivo
        self.n, self.x, self.y = read_file(input_file)
        self.distance_matrix = calculate_distance_matrix(
            self.n, self.x, self.y)

        # Define o espaço de busca para os hiperparâmetros
        self.space = [
            Integer(50, 200, name='population_size'),
            Real(0.05, 0.3, name='elite_size'),
            Real(0.001, 0.1, name='mutation_rate'),
            Real(0.05, 0.3, name='two_opt_rate'),
            Real(0.5, 0.9, name='crossover_rate')
        ]

    def objective_function(self, params):
        """
        Função objetivo para a otimização bayesiana.

        Args:
            params: Lista de valores para os hiperparâmetros (na ordem definida em self.space)

        Returns:
            Fitness do melhor indivíduo após a evolução
        """
        # Descompacta os parâmetros da lista
        population_size, elite_size, mutation_rate, two_opt_rate, crossover_rate = params

        start_time = time.time()

        if self.verbose:
            print(f"\nTestando hiperparâmetros:")
            print(f"  - População: {population_size}")
            print(f"  - Elite: {elite_size}")
            print(f"  - Taxa de mutação: {mutation_rate}")
            print(f"  - Taxa de 2-opt: {two_opt_rate}")
            print(f"  - Taxa de crossover: {crossover_rate}")

        # Configura o otimizador genético com os parâmetros atuais
        optimizer = GeneticOptimizer(
            population_size=int(population_size),
            elite_size=elite_size,
            mutation_rate=mutation_rate,
            two_opt_rate=two_opt_rate,
            crossover_rate=crossover_rate,
            use_parallel=True
        )

        # Inicializa a população
        population = [Individual(Individual.create_random_path(self.n), self.distance_matrix)
                      for _ in range(optimizer.population_size)]

        # Evolui a população
        results, _ = optimizer.evolve(population, self.n_generations)

        # Obtém o melhor fitness
        best_fitness = results[0].fitness

        if self.verbose:
            execution_time = time.time() - start_time
            print(f"  - Melhor fitness: {best_fitness}")
            print(f"  - Tempo de execução: {execution_time:.2f} segundos")

        if best_fitness < self.best_fitness:
            self.best_fitness = best_fitness
            self.best_params = {
                'population_size': int(population_size),
                'elite_size': elite_size,
                'mutation_rate': mutation_rate,
                'two_opt_rate': two_opt_rate,
                'crossover_rate': crossover_rate
            }

            if self.verbose:
                print(f"  >>> Novo melhor fitness encontrado: {best_fitness}")

        return best_fitness

    def optimize(self) -> Dict[str, Any]:
        """
        Executa a otimização bayesiana para encontrar os melhores hiperparâmetros.

        Returns:
            Dicionário com os melhores hiperparâmetros encontrados
        """
        print(
            f"Iniciando otimização bayesiana com {self.n_calls} chamadas ({self.n_random_starts} pontos aleatórios iniciais)")
        print(f"Arquivo de entrada: {self.input_file}")
        print(f"Número de cidades: {self.n}")
        print(f"Número de gerações por execução: {self.n_generations}")

        self.results = gp_minimize(
            self.objective_function,
            self.space,
            n_calls=self.n_calls,
            n_random_starts=self.n_random_starts,
            verbose=self.verbose,
            n_jobs=1  # Para evitar problemas com o multiprocessing
        )

        # Atualiza os melhores parâmetros se não foram definidos durante a otimização
        if self.best_params is None:
            self.best_params = {
                'population_size': int(self.results.x[0]),
                'elite_size': self.results.x[1],
                'mutation_rate': self.results.x[2],
                'two_opt_rate': self.results.x[3],
                'crossover_rate': self.results.x[4]
            }
            self.best_fitness = self.results.fun

        print("\nOtimização bayesiana concluída!")
        print(f"Melhores hiperparâmetros encontrados:")
        print(f"  - População: {self.best_params['population_size']}")
        print(f"  - Elite: {self.best_params['elite_size']:.4f}")
        print(f"  - Taxa de mutação: {self.best_params['mutation_rate']:.6f}")
        print(f"  - Taxa de 2-opt: {self.best_params['two_opt_rate']:.4f}")
        print(
            f"  - Taxa de crossover: {self.best_params['crossover_rate']:.4f}")
        print(f"  - Melhor fitness: {self.best_fitness}")

        return self.best_params

    def plot_results(self):
        """
        Plota os resultados da otimização bayesiana.
        """
        if self.results is None:
            print(
                "Nenhum resultado de otimização disponível. Execute optimize() primeiro.")
            return

        # Plota a convergência
        plt.figure(figsize=(10, 6))
        plot_convergence(self.results)
        plt.title("Convergência da Otimização Bayesiana")
        plt.tight_layout()
        plt.show()

        # Plota os objetivos para cada parâmetro
        fig, ax = plt.subplots(3, 2, figsize=(15, 12))
        plot_objective(self.results, dimensions=[
                       'population_size', 'elite_size'], ax=ax[0, 0])
        plot_objective(self.results, dimensions=[
                       'mutation_rate', 'two_opt_rate'], ax=ax[0, 1])
        plot_objective(self.results, dimensions=[
                       'crossover_rate', 'population_size'], ax=ax[1, 0])
        plot_objective(self.results, dimensions=[
                       'elite_size', 'mutation_rate'], ax=ax[1, 1])
        plot_objective(self.results, dimensions=[
                       'two_opt_rate', 'crossover_rate'], ax=ax[2, 0])

        # Remove o gráfico extra
        ax[2, 1].axis('off')

        plt.tight_layout()
        plt.show()

    def run_with_optimal_params(self) -> Tuple[List[Individual], float]:
        """
        Executa o algoritmo genético com os melhores parâmetros encontrados.

        Returns:
            Tupla contendo a lista de resultados e o tempo de execução
        """
        if self.best_params is None:
            print("Nenhum parâmetro ótimo disponível. Execute optimize() primeiro.")
            return None, 0

        print("\nExecutando algoritmo genético com os parâmetros ótimos...")
        inicio = time.time()

        # Configura o otimizador genético com os melhores parâmetros
        otimizador = GeneticOptimizer(
            population_size=self.best_params['population_size'],
            elite_size=self.best_params['elite_size'],
            mutation_rate=self.best_params['mutation_rate'],
            two_opt_rate=self.best_params['two_opt_rate'],
            crossover_rate=self.best_params['crossover_rate'],
            use_parallel=True
        )

        # Inicializa a população
        populacao = [Individual(Individual.create_random_path(self.n), self.distance_matrix)
                     for _ in range(otimizador.population_size)]

        # Evolui a população com mais gerações para o resultado final
        num_geracoes = 1000  # Mais gerações para a execução final
        resultados, geracao_final = otimizador.evolve(populacao, num_geracoes)

        # Calcula o tempo de execução
        fim = time.time()
        tempo_execucao = fim - inicio

        # Exibe os resultados
        melhor_individuo = resultados[0]
        print(f"Geração: {geracao_final}")
        print(f"Melhor caminho: {melhor_individuo.gene}")
        print(f"Custo: {round(melhor_individuo.fitness, 2)}")
        print(f"Tempo de execução: {tempo_execucao:.2f} segundos")

        return resultados, tempo_execucao
