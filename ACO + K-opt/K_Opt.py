from itertools import combinations
import numpy as np


class K_Opt:
    def __init__(self, graph):
        self.graph = graph

    def generate_k_opt_moves(self, solution, k):
        """
        Generates all possible k-opt moves for a given solution.

        Parameters:
            solution (list): The current solution.
            k (int): The number of edges to be swapped.

        Returns:
            list: A list of tuples, where each tuple represents a k-opt move.
        """
        n = len(solution)
        if k > n:
            return []
        indices = np.arange(n)
        return np.array(list(combinations(indices, k)))

    def reconnect_edges(self, solution, move, k):
        """
        Reconnects edges in a given solution based on a move and k value.
        Args:
            solution (list): The current solution.
            move (list): The move to be applied.
            k (int): The value of k.
        Returns:
            list: A list of new solutions after reconnecting edges.
        Raises:
            None.
        """
        if k == 2:
            return self._reconnect_2_opt(solution, move)
        elif k == 3:
            return self._reconnect_3_opt(solution, move)
        else:
            return self._reconnect_k_opt(solution, move, k)

    def _reconnect_2_opt(self, solution, move):
        i, j = sorted(move)
        return [solution[:i] + list(reversed(solution[i:j])) + solution[j:]]

    def _reconnect_3_opt(self, solution, move):
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
        # Simplificação para k > 3 usando rotações
        n = len(solution)
        move = sorted(move) + [move[0] + n]  # Fecha o ciclo
        best_solution = solution
        best_cost = self.calculate_cost(solution)

        for i in range(1, k - 1):
            new_solution = solution[:move[i]] + solution[move[i]
                :move[i + 1]][::-1] + solution[move[i + 1]:]
            new_cost = self.calculate_cost(new_solution)
            if new_cost < best_cost:
                best_solution = new_solution
                best_cost = new_cost
        return [best_solution]

    def calculate_cost(self, solution):
        """
        Calculates the cost of a given solution in the TSP problem.

        Parameters:
        - solution (list): A list representing a solution to the TSP problem.

        Returns:
        - cost (float): The cost of the given solution.
        """
        cost = np.sum(self.graph[solution[:-1], solution[1:]]
                      ) + self.graph[solution[-1], solution[0]]
        return cost

    def k_opt(self, solution, k):
        """
        Applies the k-opt heuristic to improve a given solution for the Traveling Salesman Problem (TSP).
        Parameters:
            solution (list): The initial solution to be improved.
            k (int): The number of edges to be reconnected in each move.
        Returns:
            tuple: A tuple containing the best improved solution and its corresponding cost.
        """
        best_solution = solution.copy()
        best_cost = self.calculate_cost(solution)

        for move in self.generate_k_opt_moves(solution, k):
            for new_solution in self.reconnect_edges(solution, move, k):
                new_cost = self.calculate_cost(new_solution)
                if new_cost < best_cost:
                    best_solution = new_solution
                    best_cost = new_cost

        return best_solution, best_cost
