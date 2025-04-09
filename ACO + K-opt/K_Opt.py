from itertools import combinations
import numpy as np


class K_Opt:
    """
    Implementa o algoritmo K-Opt para otimização local de rotas no Problema do Caixeiro Viajante.
    
    K-Opt é uma heurística que melhora uma solução existente substituindo k arestas
    por k outras arestas para encontrar um caminho mais curto.
    """

    def __init__(self, graph):
        """
        Inicializa a classe K_Opt.
        
        Args:
            graph (numpy.ndarray): Matriz de distâncias entre as cidades.
        """
        self.graph = graph

    def generate_k_opt_moves(self, solution, k):
        """
        Gera todos os possíveis movimentos k-opt para uma solução dada.

        Args:
            solution (list): A solução atual.
            k (int): Número de arestas a serem trocadas.

        Returns:
            numpy.ndarray: Uma lista de tuplas, onde cada tupla representa um movimento k-opt.
        """
        n = len(solution)
        if k > n:
            return []
        indices = np.arange(n)
        return np.array(list(combinations(indices, k)))

    def reconnect_edges(self, solution, move, k):
        """
        Reconecta arestas em uma solução com base em um movimento e valor de k.
        
        Args:
            solution (list): A solução atual.
            move (list): O movimento a ser aplicado.
            k (int): O valor de k (número de arestas a serem trocadas).
            
        Returns:
            list: Uma lista de novas soluções após reconectar as arestas.
        """
        if k == 2:
            return self._reconnect_2_opt(solution, move)
        elif k == 3:
            return self._reconnect_3_opt(solution, move)
        else:
            return self._reconnect_k_opt(solution, move, k)

    def _reconnect_2_opt(self, solution, move):
        """
        Implementa a reconexão de arestas para 2-opt.
        
        Args:
            solution (list): A solução atual.
            move (list): O movimento a ser aplicado.
            
        Returns:
            list: Uma lista contendo a nova solução.
        """
        i, j = sorted(move)
        return [solution[:i] + list(reversed(solution[i:j])) + solution[j:]]

    def _reconnect_3_opt(self, solution, move):
        """
        Implementa a reconexão de arestas para 3-opt.
        
        Args:
            solution (list): A solução atual.
            move (list): O movimento a ser aplicado.
            
        Returns:
            list: Uma lista de 7 possíveis reconexões.
        """
        i, j, l = sorted(move)
        a, b, c = solution[:i], solution[i:j], solution[j:l]
        d = solution[l:]
        return [
            a + b + c + d,
            a + b + c[::-1] + d,
            a + b[::-1] + c + d,
            a + b[::-1] + c[::-1] + d,
            a + c + b + d,
            a + c + b[::-1] + d,
            a + c[::-1] + b + d,
        ]

    def _reconnect_k_opt(self, solution, move, k):
        """
        Implementa a reconexão de arestas para k > 3.
        
        Args:
            solution (list): A solução atual.
            move (list): O movimento a ser aplicado.
            k (int): O valor de k.
            
        Returns:
            list: Uma lista contendo a melhor solução encontrada.
        """
        n = len(solution)
        move = sorted(move) + [move[0] + n]  # Fecha o ciclo
        best_solution = solution
        best_cost = self.calculate_cost(solution)

        for i in range(1, k - 1):
            new_solution = solution[:move[i]] + solution[move[i]:move[i + 1]][::-1] + solution[move[i + 1]:]
            new_cost = self.calculate_cost(new_solution)
            if new_cost < best_cost:
                best_solution = new_solution
                best_cost = new_cost
        return [best_solution]

    def calculate_cost(self, solution):
        """
        Calcula o custo de uma solução para o problema do TSP.

        Args:
            solution (list): Lista representando uma solução para o TSP.

        Returns:
            float: O custo da solução.
        """
        cost = np.sum(self.graph[solution[:-1], solution[1:]]) + self.graph[solution[-1], solution[0]]
        return cost

    def k_opt(self, solution, k):
        """
        Aplica a heurística k-opt para melhorar uma solução para o Problema do Caixeiro Viajante.
        
        Args:
            solution (list): A solução inicial a ser melhorada.
            k (int): Número de arestas a serem reconectadas em cada movimento.
            
        Returns:
            tuple: Uma tupla contendo a melhor solução melhorada e seu custo correspondente.
        """
        best_solution = solution.copy()
        best_cost = self.calculate_cost(solution)
        improvement_found = True
        iteration = 0
        max_iterations = min(100, len(solution) ** 2)  # Limite para problemas grandes
        
        while improvement_found and iteration < max_iterations:
            improvement_found = False
            iteration += 1
            
            for move in self.generate_k_opt_moves(solution, k):
                for new_solution in self.reconnect_edges(solution, move, k):
                    new_cost = self.calculate_cost(new_solution)
                    if new_cost < best_cost:
                        best_solution = new_solution
                        best_cost = new_cost
                        improvement_found = True
                        break
                
                if improvement_found:
                    break
            
            if improvement_found:
                solution = best_solution.copy()

        return best_solution, best_cost
