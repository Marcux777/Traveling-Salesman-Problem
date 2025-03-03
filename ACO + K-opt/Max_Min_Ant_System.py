from Ant_Colony_Optimization import ACO
import numpy as np
from K_Opt import K_Opt


class MMAS(ACO):
    def __init__(
        self,
        graph,
        num_ants,
        alpha=1.0,
        beta=2.0,
        rho=0.5,
        Q=1.0,
        tau_min=10**-3,
        tau_max=10.0,
    ):
        super().__init__(graph, num_ants, alpha=1.0, beta=2.0, rho=0.5, Q=1.0)
        self.tau_max = tau_max
        self.tau_min = tau_min
        self.pheromones = np.ones_like(self.graph) * tau_max

    def run(self, start_city, num_iterations):
        for _ in range(num_iterations):
            solutions = [
                self.generate_solutions(start_city) for _ in range(self.num_ants)
            ]
            self.update_pheromones(solutions)
            self.update_best_solution(solutions)

        return [self.best_solution, self.best_cost]

    def update_pheromones(self, solutions):
        self.pheromones *= 1 - self.rho

        costs = [self.calculate_cost(sol) for sol in solutions]

        for sol, cost in zip(solutions, costs):
            from_cities = sol[:-1]
            to_cities = sol[1:]

            self.pheromones[from_cities, to_cities] += 1.0 / cost
            self.pheromones[to_cities, from_cities] += 1.0 / cost

        self.pheromones = np.clip(self.pheromones, self.tau_min, self.tau_max)

        if self.best_solution is not None:
            best_cost = self.calculate_cost(self.best_solution)
            from_cities = self.best_solution[:-1]
            to_cities = self.best_solution[1:]
            self.pheromones[from_cities, to_cities] += self.Q / best_cost
            self.pheromones[to_cities, from_cities] += self.Q / best_cost

        self.pheromones = np.clip(self.pheromones, self.tau_min, self.tau_max)
