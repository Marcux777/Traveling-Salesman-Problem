"""
Testes unitários para o Algoritmo de Otimização por Colônia de Formigas (ACO).
"""
from ACO_Algorithm.optimization.k_opt import K_Opt
from ACO_Algorithm.aco.mmas import MMAS
from ACO_Algorithm.aco.colony import ACO
import unittest
import numpy as np
import sys
import os

# Adiciona o diretório raiz ao path para importar os módulos do projeto
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../..')))


class TestACO(unittest.TestCase):
    """Testes para o algoritmo ACO."""

    def setUp(self):
        """Configura dados para os testes."""
        # Cria uma matriz de distâncias de exemplo (grafo completo pequeno)
        self.n_cities = 5
        self.graph = np.array([
            [0, 10, 15, 20, 25],
            [10, 0, 35, 25, 30],
            [15, 35, 0, 30, 10],
            [20, 25, 30, 0, 15],
            [25, 30, 10, 15, 0]
        ])
        # Coordenadas para visualização
        self.x = np.array([0, 10, 20, 10, 0])
        self.y = np.array([0, 10, 0, -10, -20])

    def test_initialization(self):
        """Testa se o ACO é inicializado corretamente."""
        aco = ACO(self.graph, num_ants=10, alpha=1.0, beta=2.0, rho=0.5)

        self.assertEqual(aco.num_ants, 10)
        self.assertEqual(aco.num_cities, self.n_cities)
        self.assertEqual(aco.alpha, 1.0)
        self.assertEqual(aco.beta, 2.0)
        self.assertEqual(aco.rho, 0.5)

        # Verifica se a matriz de feromônios foi inicializada
        self.assertEqual(aco.pheromone.shape, (self.n_cities, self.n_cities))
        # Todos os valores devem ser positivos
        self.assertTrue(np.all(aco.pheromone > 0))

    def test_solution_validity(self):
        """Testa se o ACO gera soluções válidas."""
        aco = ACO(self.graph, num_ants=5, alpha=1.0, beta=2.0, rho=0.5)
        solution, cost = aco.run(start_city=0, max_iterations=10)

        # Verifica se a solução contém todas as cidades
        # Alterado para corresponder à implementação atual
        self.assertEqual(len(solution), self.n_cities)

        # Verificar se a solução tem todas as cidades sem repetição (exceto a primeira)
        # Todas as cidades devem ser visitadas exatamente uma vez
        self.assertEqual(len(set(solution)), self.n_cities)

        # Verifica se o custo é calculado corretamente
        calculated_cost = aco.calculate_cost(solution)
        self.assertAlmostEqual(calculated_cost, cost)

    def test_mmas_initialization(self):
        """Testa se o MMAS é inicializado corretamente."""
        # Usar float em vez de int para distances para evitar o erro de conversão para infinito
        mmas = MMAS(self.graph.astype(float), num_ants=10,
                    alpha=1.0, beta=2.0, rho=0.5, p_best=0.05)

        self.assertEqual(mmas.num_ants, 10)
        self.assertEqual(mmas.num_cities, self.n_cities)

        # Verifica se os limites de feromônio foram inicializados
        self.assertTrue(hasattr(mmas, 'max_pheromone'))
        self.assertTrue(hasattr(mmas, 'min_pheromone'))
        self.assertTrue(mmas.max_pheromone > mmas.min_pheromone)

    def test_k_opt_improvement(self):
        """Testa se o K-Opt melhora uma solução."""
        # Cria uma solução não-ótima
        solution = [0, 1, 2, 3, 4, 0]
        k_opt = K_Opt(self.graph)

        # Aplica 2-opt usando o método k_opt em vez de optimize
        improved_solution, improved_cost = k_opt.k_opt(solution, k=2)

        # Verifica se a solução melhorada é válida
        self.assertEqual(len(improved_solution), len(solution))

        # Calcula o custo da solução original para comparação
        original_cost = k_opt.calculate_cost(solution)

        # A solução melhorada deve ter um custo menor ou igual
        self.assertTrue(improved_cost <= original_cost)

    def test_pheromone_update(self):
        """Testa se a atualização de feromônios funciona como esperado."""
        aco = ACO(self.graph, num_ants=5, alpha=1.0, beta=2.0, rho=0.5)

        # Salva o estado inicial dos feromônios
        initial_pheromone = aco.pheromone.copy()

        # Executa algumas iterações
        aco.run(start_city=0, max_iterations=5)

        # Verifica se os feromônios foram atualizados
        self.assertFalse(np.array_equal(aco.pheromone, initial_pheromone))

        # Todos os valores de feromônio devem ser positivos
        self.assertTrue(np.all(aco.pheromone > 0))


if __name__ == '__main__':
    unittest.main()
