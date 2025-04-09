"""
Módulo que implementa o Algoritmo de Otimização por Colônia de Formigas (ACO) 
para resolver o Problema do Caixeiro Viajante.
"""

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
        prob = (pheromone ** self.alpha) * (visibility ** self.beta)
        
        # Normaliza as probabilidades
        prob = prob / (np.sum(prob) + 1e-10)
        
        # Seleção usando roleta de probabilidade
        next_city = np.random.choice(np.where(unvisited)[0], p=prob)
        
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
