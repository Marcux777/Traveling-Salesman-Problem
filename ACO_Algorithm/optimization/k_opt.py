"""
Módulo que implementa o algoritmo K-Opt para otimização local de rotas.
"""

import numpy as np
from itertools import combinations


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

        # Converte a solução para uma lista de Python para evitar problemas de broadcast
        solution = list(solution)

        # Verifica os casos de borda para evitar problemas de broadcasting
        if i == 0 and j == len(solution):
            # Se i=0 e j=len(solution), inverte toda a solução
            return [list(reversed(solution))]
        elif i == 0:
            # Se i=0, não há necessidade de concatenar uma parte vazia no início
            part1 = list(reversed(solution[:j]))
            part2 = list(solution[j:])
            return [part1 + part2]
        elif j == len(solution):
            # Se j=len(solution), não há necessidade de concatenar uma parte vazia no final
            part1 = list(solution[:i])
            part2 = list(reversed(solution[i:]))
            return [part1 + part2]
        else:
            # Caso normal
            part1 = list(solution[:i])
            part2 = list(reversed(solution[i:j]))
            part3 = list(solution[j:])
            return [part1 + part2 + part3]

    def _reconnect_3_opt(self, solution, move):
        """
        Implementa a reconexão de arestas para 3-opt.

        Args:
            solution (list): A solução atual.
            move (list): O movimento a ser aplicado.

        Returns:
            list: Uma lista de 7 possíveis reconexões.
        """
        # Converte a solução para uma lista Python
        solution = list(solution)

        i, j, l = sorted(move)
        # Cria segmentos como listas comuns
        a = list(solution[:i])
        b = list(solution[i:j])
        c = list(solution[j:l])
        d = list(solution[l:])

        # Cria todas as possíveis reconexões
        return [
            a + b + c + d,
            a + b + list(reversed(c)) + d,
            a + list(reversed(b)) + c + d,
            a + list(reversed(b)) + list(reversed(c)) + d,
            a + c + b + d,
            a + c + list(reversed(b)) + d,
            a + list(reversed(c)) + b + d,
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
        # Converte a solução para uma lista Python
        solution = list(solution)

        n = len(solution)
        move = sorted(move) + [move[0] + n]  # Fecha o ciclo
        best_solution = solution.copy()
        best_cost = self.calculate_cost(solution)

        for i in range(1, k - 1):
            # Cria cada parte como lista
            part1 = list(solution[:move[i]])
            part2 = list(reversed(solution[move[i]:move[i + 1]]))
            part3 = list(solution[move[i + 1]:])

            # Combina as partes
            new_solution = part1 + part2 + part3
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
        # Verifica se algum índice está fora dos limites
        valid_solution = [i for i in solution if 0 <= i < len(self.graph)]

        # Se a solução resultante for muito curta, retorna um custo muito alto
        if len(valid_solution) < 2:
            return float('inf')

        # Calcula o custo apenas com índices válidos
        cost = np.sum(self.graph[valid_solution[:-1], valid_solution[1:]])

        # Adiciona o custo de retorno à cidade inicial, se possível
        if len(valid_solution) > 1:
            cost += self.graph[valid_solution[-1], valid_solution[0]]

        return cost

    def k_opt(self, solution, k, max_iterations=None):
        """
        Aplica a heurística k-opt para melhorar uma solução para o Problema do Caixeiro Viajante.

        Args:
            solution (list): A solução inicial a ser melhorada.
            k (int): Número de arestas a serem reconectadas em cada movimento.
            max_iterations (int, optional): Número máximo de iterações. Se None,
                                          será calculado com base no tamanho da solução.

        Returns:
            tuple: Uma tupla contendo a melhor solução melhorada e seu custo correspondente.
        """
        best_solution = solution.copy()
        best_cost = self.calculate_cost(solution)
        improvement_found = True
        iteration = 0

        # Limite para problemas grandes
        if max_iterations is None:
            max_iterations = min(100, len(solution) ** 2)

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
