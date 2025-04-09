"""
Módulo que implementa o Sistema de Formigas Max-Min (MMAS), uma variante do
Algoritmo de Otimização por Colônia de Formigas (ACO) para o Problema do Caixeiro Viajante.
"""

import numpy as np
from Ant_Colony_Optimization import ACO


class MMAS(ACO):
    """
    Implementação do Sistema de Formigas Max-Min (MMAS).

    MMAS é uma variante do algoritmo ACO que limita os valores de feromônio
    em um intervalo [min, max] para evitar convergência prematura.
    """

    def __init__(self, graph, num_ants=10, alpha=1.0, beta=2.0, rho=0.1, q=1.0,
                 p_best=0.05, stagnation_limit=10):
        """
        Inicializa o algoritmo MMAS.

        Args:
            graph (numpy.ndarray): Matriz de distâncias entre as cidades.
            num_ants (int): Número de formigas na colônia.
            alpha (float): Importância dos feromônios na decisão.
            beta (float): Importância da heurística de distância na decisão.
            rho (float): Taxa de evaporação dos feromônios (0-1).
            q (float): Constante para atualização dos feromônios.
            p_best (float): Probabilidade de escolher o melhor caminho (para cálculo de limites).
            stagnation_limit (int): Número de iterações sem melhoria para reinicializar.
        """
        # Inicializa a classe pai (ACO)
        super().__init__(graph, num_ants, alpha, beta, rho, q)

        self.p_best = p_best
        self.stagnation_limit = stagnation_limit
        self.stagnation_count = 0

        # Inicializa com valores máximos os feromônios
        self._initialize_pheromone_limits()
        self.pheromone = np.ones(
            (self.num_cities, self.num_cities)) * self.max_pheromone

    def _initialize_pheromone_limits(self):
        """
        Inicializa os limites máximo e mínimo para os valores de feromônio.
        """
        # Usamos uma heurística gulosa para estimar um valor inicial
        # Começamos de uma cidade aleatória
        start = np.random.randint(0, self.num_cities)

        # Construímos uma solução gulosa
        route = [start]
        while len(route) < self.num_cities:
            current = route[-1]
            # Encontra a cidade mais próxima que ainda não foi visitada
            distances = self.graph[current].copy()
            for city in route:
                distances[city] = np.inf
            next_city = np.argmin(distances)
            route.append(next_city)

        # Calcula o custo da solução gulosa
        greedy_cost = self.calculate_cost(route)

        # Estimamos os limites dos feromônios com base nessa solução
        self.max_pheromone = 1.0 / (self.rho * greedy_cost)
        self.min_pheromone = self.max_pheromone * (1.0 - np.power(self.p_best, 1.0 / self.num_cities)) / \
            ((self.num_cities / 2 - 1) * np.power(self.p_best, 1.0 / self.num_cities))

    def update_pheromones(self, routes, costs):
        """
        Atualiza a matriz de feromônios usando a abordagem Max-Min.

        Args:
            routes (numpy.ndarray): Matriz contendo as rotas das formigas.
            costs (numpy.ndarray): Vetor contendo os custos das rotas.
        """
        # Evapora todos os feromônios
        self.pheromone *= (1 - self.rho)

        # No MMAS, apenas a melhor formiga (global ou da iteração) deposita feromônio
        best_idx = np.argmin(costs)
        best_route = routes[best_idx]
        best_cost = costs[best_idx]

        # Adiciona feromônio às arestas usadas pela melhor formiga
        for i in range(self.num_cities - 1):
            self.pheromone[best_route[i],
                           best_route[i+1]] += self.q / best_cost

        # Adiciona feromônio à aresta de retorno
        self.pheromone[best_route[-1], best_route[0]] += self.q / best_cost

        # Aplica os limites max-min
        self.pheromone = np.clip(
            self.pheromone, self.min_pheromone, self.max_pheromone)

    def run(self, start_city=0, max_iterations=100):
        """
        Executa o algoritmo MMAS por um número determinado de iterações.

        Args:
            start_city (int): Cidade inicial para todas as formigas.
            max_iterations (int): Número máximo de iterações.

        Returns:
            tuple: Uma tupla contendo a melhor rota encontrada e seu custo.
        """
        # Inicializa a melhor solução
        best_solution = None
        best_cost = float('inf')
        best_iteration = 0

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
                best_iteration = iteration
                self.stagnation_count = 0
            else:
                self.stagnation_count += 1

            # Recalcula os limites de feromônio se necessário
            if iteration - best_iteration > 50:
                self._initialize_pheromone_limits()

            # Reinicializa os feromônios se houver estagnação
            if self.stagnation_count >= self.stagnation_limit:
                print(
                    f"Reinicializando feromônios na iteração {iteration} por estagnação")
                self.pheromone = np.ones(
                    (self.num_cities, self.num_cities)) * self.max_pheromone
                self.stagnation_count = 0

            # Atualiza a matriz de feromônios
            self.update_pheromones(routes, costs)

        return best_solution, best_cost
