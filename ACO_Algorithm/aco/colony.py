"""
Módulo que implementa o Algoritmo de Otimização por Colônia de Formigas (ACO)
para resolver o Problema do Caixeiro Viajante.
"""

from ..utils.visualization import plot_tsp_solution, plot_convergence
from ..optimization.k_opt import K_Opt
from ..aco.max_min_ant_system import MMAS
from ..aco.ant_colony import ACO
from multiprocessing import Pool, cpu_count
import time
import numpy as np
import random


class ACO:
    """
    Implementação do algoritmo de Otimização por Colônia de Formigas.

    Esta classe oferece uma implementação do algoritmo ACO padrão para resolver
    o Problema do Caixeiro Viajante, baseado na simulação do comportamento de formigas
    procurando o caminho mais curto usando feromônios.
    """

    def __init__(self, graph, num_ants=10, alpha=1.0, beta=1.0, rho=0.5, q=1.0):
        """
        Inicializa o algoritmo ACO.

        Args:
            graph (numpy.ndarray): Matriz de distâncias entre as cidades.
            num_ants (int): Número de formigas na colônia.
            alpha (float): Importância dos feromônios na decisão.
            beta (float): Importância da heurística de distância na decisão.
            rho (float): Taxa de evaporação dos feromônios (0-1).
            q (float): Constante para atualização dos feromônios.
        """
        self.graph = graph  # Matriz de distâncias
        self.num_ants = num_ants  # Número de formigas
        self.num_cities = graph.shape[0]  # Número de cidades
        self.alpha = alpha  # Importância do feromônio
        self.beta = beta    # Importância da heurística
        self.rho = rho      # Taxa de evaporação
        self.q = q          # Constante para cálculo da deposição de feromônio

        # Inicializa a matriz de feromônios com uma pequena quantidade
        self.pheromone = np.ones((self.num_cities, self.num_cities)) * 0.1

        # Calcula a visibilidade (1/distância)
        # Usamos máx(0.1, dist) para evitar divisão por zero
        self.visibility = 1.0 / np.maximum(0.1, self.graph)
        np.fill_diagonal(self.visibility, 0.0)  # Zero na diagonal

    def select_next_city(self, ant, visited):
        """
        Seleciona a próxima cidade para uma formiga usando a regra de probabilidade.

        Args:
            ant (int): Cidade atual da formiga.
            visited (list): Lista de cidades já visitadas.

        Returns:
            int: Índice da próxima cidade a ser visitada.
        """
        # Lista de cidades não visitadas
        unvisited = np.ones(self.num_cities, dtype=bool)
        unvisited[visited] = False

        # Se não houver cidades não visitadas, retorna -1
        if not np.any(unvisited):
            return -1

        # Cálculo da probabilidade para cada cidade não visitada
        pheromone = self.pheromone[ant, unvisited]
        visibility = self.visibility[ant, unvisited]

        # Aplica a fórmula de probabilidade do ACO
        numerator = (pheromone ** self.alpha) * (visibility ** self.beta)

        # Verifica se todos os valores são zero
        if np.sum(numerator) == 0:
            # Se todos forem zero, usa probabilidade uniforme
            prob = np.ones_like(numerator) / len(numerator)
        else:
            # Normaliza as probabilidades
            prob = numerator / np.sum(numerator)

        # Verifica se as probabilidades somam 1 (com tolerância para erro de ponto flutuante)
        sum_prob = np.sum(prob)
        if not np.isclose(sum_prob, 1.0):
            prob = prob / sum_prob  # Renormaliza se necessário

        # Seleção usando roleta de probabilidade
        try:
            next_city = np.random.choice(np.where(unvisited)[0], p=prob)
        except ValueError:
            # Se ainda houver problemas com as probabilidades, escolhe aleatoriamente
            unvisited_indices = np.where(unvisited)[0]
            next_city = np.random.choice(unvisited_indices)

        return next_city

    def construct_solutions(self, start_city=0):
        """
        Constrói soluções para todas as formigas.

        Args:
            start_city (int): Cidade inicial para todas as formigas.

        Returns:
            tuple: Uma tupla contendo as rotas construídas pelas formigas
                  e seus custos correspondentes.
        """
        # Inicializa rotas e custos
        routes = np.zeros((self.num_ants, self.num_cities), dtype=int)
        costs = np.zeros(self.num_ants)

        # Para cada formiga, constrói uma solução
        for ant in range(self.num_ants):
            # Inicia na cidade especificada
            current_city = start_city
            visited = [current_city]

            # Constrói a rota visitando todas as cidades
            while len(visited) < self.num_cities:
                next_city = self.select_next_city(current_city, visited)
                visited.append(next_city)
                current_city = next_city

            # Armazena a rota e calcula seu custo
            routes[ant] = visited
            costs[ant] = self.calculate_cost(visited)

        return routes, costs

    def calculate_cost(self, route):
        """
        Calcula o custo de uma rota.

        Args:
            route (list): Lista de cidades representando uma rota.

        Returns:
            float: Custo total da rota.
        """
        # Soma das distâncias entre cidades consecutivas
        cost = np.sum(self.graph[route[:-1], route[1:]])

        # Adiciona o retorno à cidade inicial
        cost += self.graph[route[-1], route[0]]

        return cost

    def update_pheromones(self, routes, costs):
        """
        Atualiza a matriz de feromônios com base nas rotas construídas.

        Args:
            routes (numpy.ndarray): Matriz contendo as rotas das formigas.
            costs (numpy.ndarray): Vetor contendo os custos das rotas.
        """
        # Evaporação dos feromônios
        self.pheromone *= (1 - self.rho)

        # Deposição de feromônios para cada formiga
        for ant in range(self.num_ants):
            route = routes[ant]
            cost = costs[ant]

            # Adiciona feromônio às arestas usadas pela formiga
            for i in range(self.num_cities - 1):
                self.pheromone[route[i], route[i + 1]] += self.q / cost

            # Adiciona feromônio à aresta de retorno
            self.pheromone[route[-1], route[0]] += self.q / cost

    def run(self, start_city=0, max_iterations=100):
        """
        Executa o algoritmo ACO por um número determinado de iterações.

        Args:
            start_city (int): Cidade inicial para todas as formigas.
            max_iterations (int): Número máximo de iterações.

        Returns:
            tuple: Uma tupla contendo a melhor rota encontrada e seu custo.
        """
        # Inicializa a melhor solução
        best_solution = None
        best_cost = float('inf')

        # Executa o algoritmo por max_iterations iterações
        for iteration in range(max_iterations):
            # Constrói soluções para todas as formigas
            routes, costs = self.construct_solutions(start_city)

            # Atualiza a melhor solução, se necessário
            current_best_idx = np.argmin(costs)
            current_best_cost = costs[current_best_idx]

            if current_best_cost < best_cost:
                best_cost = current_best_cost
                best_solution = routes[current_best_idx].copy()

            # Atualiza a matriz de feromônios
            self.update_pheromones(routes, costs)

        return best_solution, best_cost


"""
Módulo que implementa funções para execução paralela do Algoritmo de Colônia de Formigas (ACO).
"""


def run_aco_single(params):
    """
    Executa uma única instância do algoritmo ACO.

    Args:
        params (tuple): Uma tupla com os parâmetros necessários:
                       (graph, num_ants, alpha, beta, rho, start_city, max_iterations, use_mmas)

    Returns:
        tuple: Uma tupla contendo a melhor solução e seu custo.
    """
    graph, num_ants, alpha, beta, rho, start_city, max_iterations, use_mmas = params

    if use_mmas:
        algorithm = MMAS(graph, num_ants=num_ants,
                         alpha=alpha, beta=beta, rho=rho)
    else:
        algorithm = ACO(graph, num_ants=num_ants,
                        alpha=alpha, beta=beta, rho=rho)

    best_solution, best_cost = algorithm.run(start_city, max_iterations)
    return best_solution, best_cost


def run_parallel_aco(graph, num_ants, alpha, beta, rho, start_city, max_iterations,
                     use_mmas=False, num_processes=None):
    """
    Executa múltiplas instâncias do algoritmo ACO em paralelo.

    Args:
        graph (numpy.ndarray): Matriz de distâncias entre as cidades.
        num_ants (int): Número de formigas.
        alpha (float): Importância do feromônio.
        beta (float): Importância da heurística.
        rho (float): Taxa de evaporação do feromônio.
        start_city (int): Cidade inicial.
        max_iterations (int): Número máximo de iterações.
        use_mmas (bool): Se deve usar Max-Min Ant System em vez do ACO padrão.
        num_processes (int, optional): Número de processos paralelos.
                                      Se None, usa o número de CPUs disponíveis.

    Returns:
        tuple: Uma tupla contendo a melhor solução e seu custo.
    """
    if num_processes is None:
        num_processes = cpu_count()

    params = (graph, num_ants, alpha, beta, rho,
              start_city, max_iterations, use_mmas)
    params_list = [params] * num_processes

    with Pool(num_processes) as pool:
        results = pool.map(run_aco_single, params_list)

    return min(results, key=lambda x: x[1])


def run_standard_aco(graph, x, y, num_ants=30, alpha=0.8, beta=0.8, rho=0.9,
                     start_city=0, max_iterations=100, apply_k_opt=True, k_value=2,
                     use_mmas=False, plot_results=True):
    """
    Executa o algoritmo ACO padrão com paralelização e mostra os resultados.

    Args:
        graph (numpy.ndarray): Matriz de distâncias entre as cidades.
        x (list): Lista de coordenadas x das cidades.
        y (list): Lista de coordenadas y das cidades.
        num_ants (int): Número de formigas.
        alpha (float): Importância do feromônio.
        beta (float): Importância da heurística.
        rho (float): Taxa de evaporação do feromônio.
        start_city (int): Cidade inicial.
        max_iterations (int): Número máximo de iterações.
        apply_k_opt (bool): Se deve aplicar a heurística K-Opt.
        k_value (int): Valor de k para K-Opt.
        use_mmas (bool): Se deve usar Max-Min Ant System em vez do ACO padrão.
        plot_results (bool): Se deve plotar os resultados.

    Returns:
        tuple: Uma tupla contendo a melhor solução e seu custo.
    """
    algorithm_name = "MMAS" if use_mmas else "ACO padrão"
    print(
        f"Executando {algorithm_name} com {num_ants} formigas, alpha={alpha}, beta={beta}, rho={rho}")
    print(f"Número de cidades: {len(x)}")

    start_time = time.time()

    # Para coletar dados de convergência, vamos executar uma instância única também
    if use_mmas:
        algorithm = MMAS(graph, num_ants=num_ants,
                         alpha=alpha, beta=beta, rho=rho)
    else:
        algorithm = ACO(graph, num_ants=num_ants,
                        alpha=alpha, beta=beta, rho=rho)

    single_solution, single_cost = algorithm.run(start_city, max_iterations)

    # Executa ACO em paralelo para obter múltiplas soluções
    best_solution, best_cost = run_parallel_aco(
        graph, num_ants, alpha, beta, rho, start_city, max_iterations, use_mmas
    )

    # Usa a melhor solução entre a execução única e as paralelas
    if single_cost < best_cost:
        best_solution, best_cost = single_solution, single_cost

    # Aplica K-Opt na melhor solução
    if apply_k_opt:
        print(f"Aplicando {k_value}-opt na melhor solução...")
        k_opt = K_Opt(graph)
        improved_solution, improved_cost = k_opt.k_opt(best_solution, k_value)

        if improved_cost < best_cost:
            print(
                f"K-Opt melhorou a solução: {best_cost:.2f} -> {improved_cost:.2f}")
            best_solution = improved_solution
            best_cost = improved_cost

    end_time = time.time()
    execution_time = end_time - start_time

    print("Melhor solução:", best_solution)
    print("Custo:", best_cost)
    print("Tempo de execução:", execution_time, "segundos")

    # Plotar a solução e dados de convergência
    if plot_results:
        plot_tsp_solution(best_solution, x, y, best_cost)

        # Plota gráfico de convergência se disponível
        iterations, costs = algorithm.get_convergence_data()
        if iterations and costs:
            plot_convergence(iterations, costs,
                             f"Convergência do {algorithm_name}")

    return best_solution, best_cost
