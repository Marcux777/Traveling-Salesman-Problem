"""
Módulo que implementa otimização bayesiana para os hiperparâmetros da Otimização por Colônia de Formigas.
"""

import time
import numpy as np
from typing import Dict, Any, List, Tuple
import matplotlib.pyplot as plt
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.plots import plot_convergence, plot_objective

from ACO_Algorithm.aco.colony import ACO
from ACO_Algorithm.optimization.k_opt import K_Opt
from ACO_Algorithm.utils.visualization import plot_tsp_solution


class BayesianOptimizerACO:
    """
    Classe que implementa otimização bayesiana para os hiperparâmetros de ACO.
    """

    def __init__(self,
                 graph: np.ndarray,
                 x: List[float],
                 y: List[float],
                 n_calls: int = 20,
                 n_random_starts: int = 5,
                 max_iterations: int = 50,
                 start_city: int = 0,
                 verbose: bool = True,
                 use_k_opt: bool = True,
                 k_value: int = 2):
        """
        Inicializa o otimizador bayesiano para ACO.

        Args:
            graph: Matriz de distâncias entre as cidades
            x: Coordenadas x das cidades
            y: Coordenadas y das cidades
            n_calls: Número de chamadas para a otimização bayesiana
            n_random_starts: Número de pontos aleatórios iniciais
            max_iterations: Número de iterações para cada execução de ACO
            start_city: Cidade inicial para a construção das rotas
            verbose: Se deve exibir informações de progresso
            use_k_opt: Se deve aplicar a heurística k-opt nas melhores soluções
            k_value: Valor de k para o algoritmo k-opt
        """
        self.graph = graph
        self.x = x
        self.y = y
        self.n_calls = n_calls
        self.n_random_starts = n_random_starts
        self.max_iterations = max_iterations
        self.start_city = start_city
        self.verbose = verbose
        self.use_k_opt = use_k_opt
        self.k_value = k_value
        self.best_params = None
        self.best_cost = float('inf')
        self.results = None

        # Define o espaço de busca para os hiperparâmetros
        self._define_search_space()

    def _define_search_space(self):
        """
        Define o espaço de busca para os hiperparâmetros do ACO.
        """
        self.space = [
            Integer(5, 50, name='num_ants'),
            Real(0.1, 2.0, name='alpha'),
            Real(0.1, 5.0, name='beta'),
            Real(0.1, 0.99, name='rho')
        ]

    def _run_aco_with_params(self, params: List) -> Tuple[List[int], float]:
        """
        Executa o ACO com um conjunto específico de parâmetros.

        Args:
            params: Lista de valores para os hiperparâmetros

        Returns:
            Tupla contendo a melhor solução e seu custo
        """
        # Descompacta os parâmetros da lista
        num_ants, alpha, beta, rho = params

        # Configura o ACO com os parâmetros atuais
        aco = ACO(
            self.graph,
            num_ants=int(num_ants),
            alpha=alpha,
            beta=beta,
            rho=rho
        )

        # Executa o ACO
        best_solution, best_cost = aco.run(
            self.start_city, self.max_iterations)

        return best_solution, best_cost

    def _apply_k_opt(self, solution: List[int], cost: float) -> Tuple[List[int], float]:
        """
        Aplica a heurística K-Opt em uma solução.

        Args:
            solution: Solução a ser melhorada
            cost: Custo da solução original

        Returns:
            Tupla contendo a solução melhorada e seu custo
        """
        if not self.use_k_opt:
            return solution, cost

        k_opt = K_Opt(self.graph)
        improved_solution, improved_cost = k_opt.k_opt(solution, self.k_value)

        return improved_solution, improved_cost

    def objective_function(self, params: List) -> float:
        """
        Função objetivo para a otimização bayesiana.

        Args:
            params: Lista de valores para os hiperparâmetros (na ordem definida em self.space)

        Returns:
            Custo da melhor solução após a execução de ACO
        """
        start_time = time.time()

        if self.verbose:
            self._print_params(params)

        # Executa o ACO com os parâmetros atuais
        best_solution, best_cost = self._run_aco_with_params(params)

        # Aplica K-Opt na melhor solução encontrada, se solicitado
        if self.use_k_opt:
            improved_solution, improved_cost = self._apply_k_opt(
                best_solution, best_cost)

            if improved_cost < best_cost:
                best_solution = improved_solution
                best_cost = improved_cost

        if self.verbose:
            execution_time = time.time() - start_time
            print(f"  - Melhor custo: {best_cost:.2f}")
            print(f"  - Tempo de execução: {execution_time:.2f} segundos")

        # Atualiza os melhores parâmetros se necessário
        self._update_best_params(params, best_cost)

        return best_cost

    def _print_params(self, params: List):
        """
        Imprime os parâmetros atuais sendo testados.

        Args:
            params: Lista de valores para os hiperparâmetros
        """
        num_ants, alpha, beta, rho = params
        print(f"\nTestando hiperparâmetros:")
        print(f"  - Número de formigas: {num_ants}")
        print(f"  - Alpha (importância do feromônio): {alpha:.2f}")
        print(f"  - Beta (importância da heurística): {beta:.2f}")
        print(f"  - Rho (taxa de evaporação): {rho:.2f}")

    def _update_best_params(self, params: List, cost: float):
        """
        Atualiza os melhores parâmetros se o custo atual for melhor.

        Args:
            params: Lista de valores para os hiperparâmetros
            cost: Custo da solução atual
        """
        if cost < self.best_cost:
            num_ants, alpha, beta, rho = params
            self.best_cost = cost
            self.best_params = {
                'num_ants': int(num_ants),
                'alpha': alpha,
                'beta': beta,
                'rho': rho
            }

            if self.verbose:
                print(f"  >>> Novo melhor custo encontrado: {cost:.2f}")

    def optimize(self) -> Dict[str, Any]:
        """
        Executa a otimização bayesiana para encontrar os melhores hiperparâmetros.

        Returns:
            Dicionário com os melhores hiperparâmetros encontrados
        """
        print(
            f"Iniciando otimização bayesiana com {self.n_calls} chamadas ({self.n_random_starts} pontos aleatórios iniciais)")
        print(f"Número de cidades: {len(self.x)}")
        print(f"Número de iterações por execução: {self.max_iterations}")

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
            self._update_best_params(
                [self.results.x[0], self.results.x[1],
                    self.results.x[2], self.results.x[3]],
                self.results.fun
            )

        self._print_optimization_results()

        return self.best_params

    def _print_optimization_results(self):
        """
        Imprime os resultados da otimização bayesiana.
        """
        print("\nOtimização bayesiana concluída!")
        print(f"Melhores hiperparâmetros encontrados:")
        print(f"  - Número de formigas: {self.best_params['num_ants']}")
        print(
            f"  - Alpha (importância do feromônio): {self.best_params['alpha']:.4f}")
        print(
            f"  - Beta (importância da heurística): {self.best_params['beta']:.4f}")
        print(f"  - Rho (taxa de evaporação): {self.best_params['rho']:.4f}")
        print(f"  - Melhor custo: {self.best_cost:.4f}")

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
        plt.title("Convergência da Otimização Bayesiana para ACO")
        plt.tight_layout()
        plt.show()

        # Plota os objetivos para cada parâmetro
        try:
            # Obtém os nomes das dimensões
            dimension_names = [dim.name for dim in self.space]

            # Plota as dimensões duas a duas
            for i in range(len(dimension_names)):
                for j in range(i+1, len(dimension_names)):
                    plt.figure(figsize=(10, 6))
                    plot_objective(self.results, dimensions=[i, j])
                    plt.title(
                        f"Otimização de {dimension_names[i]} vs {dimension_names[j]}")
                    plt.tight_layout()
                    plt.show()
        except Exception as e:
            print(f"Erro ao plotar objetivos: {e}")
            print("Continuando com a execução...")

    def run_with_optimal_params(self) -> Tuple[List[int], float]:
        """
        Executa o ACO com os melhores parâmetros encontrados.

        Returns:
            Tupla contendo a melhor solução e seu custo
        """
        if self.best_params is None:
            print("Nenhum parâmetro ótimo disponível. Execute optimize() primeiro.")
            return [], float('inf')

        print("\nExecutando ACO com os parâmetros ótimos...")
        inicio = time.time()

        # Configura o ACO com os melhores parâmetros
        aco = ACO(
            self.graph,
            num_ants=self.best_params['num_ants'],
            alpha=self.best_params['alpha'],
            beta=self.best_params['beta'],
            rho=self.best_params['rho']
        )

        # Executa o ACO com mais iterações para o resultado final
        num_iteracoes = 200  # Mais iterações para a execução final
        best_solution, best_cost = aco.run(self.start_city, num_iteracoes)

        # Aplica K-Opt na melhor solução encontrada, se solicitado
        if self.use_k_opt:
            print("Aplicando K-Opt na melhor solução...")
            improved_solution, improved_cost = self._apply_k_opt(
                best_solution, best_cost)

            if improved_cost < best_cost:
                best_solution = improved_solution
                best_cost = improved_cost
                print(f"K-Opt melhorou a solução: {best_cost:.2f}")

        # Calcula o tempo de execução
        fim = time.time()
        tempo_execucao = fim - inicio

        # Exibe os resultados
        print(f"Melhor solução: {best_solution}")
        print(f"Custo: {best_cost:.2f}")
        print(f"Tempo de execução: {tempo_execucao:.2f} segundos")

        # Plota a solução
        plot_tsp_solution(best_solution, self.x, self.y, best_cost)

        return best_solution, best_cost


def run_bayesian_optimization(graph, x, y, n_calls=20, n_random_starts=5,
                              max_iterations=50, start_city=0, k_value=2):
    """
    Executa otimização bayesiana para encontrar os melhores hiperparâmetros para ACO.

    Args:
        graph: Matriz de distâncias
        x, y: Coordenadas das cidades
        n_calls: Número de chamadas para a otimização bayesiana
        n_random_starts: Número de pontos aleatórios iniciais
        max_iterations: Número de iterações para cada execução de ACO
        start_city: Cidade inicial
        k_value: Valor de k para K-Opt

    Returns:
        A melhor solução encontrada, seu custo e os melhores parâmetros
    """
    # Configurar o otimizador bayesiano
    bayesian_opt = BayesianOptimizerACO(
        graph=graph,
        x=x,
        y=y,
        n_calls=n_calls,
        n_random_starts=n_random_starts,
        max_iterations=max_iterations,
        start_city=start_city,
        verbose=True,
        use_k_opt=True,
        k_value=k_value
    )

    # Executar a otimização
    best_params = bayesian_opt.optimize()

    # Plotar resultados da otimização
    bayesian_opt.plot_results()

    # Executar o ACO com os melhores parâmetros
    best_solution, best_cost = bayesian_opt.run_with_optimal_params()

    return best_solution, best_cost, best_params
