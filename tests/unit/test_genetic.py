"""
Testes unitários para o Algoritmo Genético (GA).
"""
from Genetic_Algorithm.genetic.utils import calculate_distance_matrix
from Genetic_Algorithm.genetic.optimizer import GeneticOptimizer
from Genetic_Algorithm.genetic.individual import Individual
import unittest
import numpy as np
import sys
import os

# Adiciona o diretório raiz ao path para importar os módulos do projeto
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../..')))


class TestGeneticAlgorithm(unittest.TestCase):
    """Testes para o algoritmo Genético."""

    def setUp(self):
        """Configura dados para os testes."""
        # Configura coordenadas de teste
        self.n = 5
        self.x = np.array([0, 10, 20, 10, 0])
        self.y = np.array([0, 10, 0, -10, -20])

        # Calcula matriz de distâncias
        self.distance_matrix = calculate_distance_matrix(
            self.n, self.x, self.y)

        # Cria um gene de exemplo para usar nos testes (índices começam em 1 na implementação atual)
        self.sample_gene = np.array([1, 2, 3, 4, 5])

    def test_individual_initialization(self):
        """Testa se o Indivíduo é inicializado corretamente."""
        individual = Individual(self.sample_gene, self.distance_matrix)

        # Verificar se o gene tem o tamanho correto
        self.assertEqual(len(individual.gene), self.n)

        # Verificar se todas as cidades estão presentes no gene
        self.assertEqual(set(individual.gene), set(range(1, self.n + 1)))

        # Verificar se o cálculo de fitness funciona
        self.assertTrue(individual.fitness > 0)

    def test_crossover(self):
        """Testa o operador de crossover."""
        parent1 = Individual(np.array([1, 2, 3, 4, 5]), self.distance_matrix)
        parent2 = Individual(np.array([5, 4, 3, 2, 1]), self.distance_matrix)

        # Realiza crossover
        child = parent1.crossover(parent2)

        # Verificar se o filho tem o tamanho correto
        self.assertEqual(len(child.gene), self.n)

        # Verificar se todas as cidades estão presentes no gene do filho
        self.assertEqual(set(child.gene), set(range(1, self.n + 1)))

        # Verificar se o filho tem genes de ambos os pais
        self.assertTrue(
            np.array_equal(child.gene, parent1.gene) == False and
            np.array_equal(child.gene, parent2.gene) == False
        )

    def test_mutation(self):
        """Testa o operador de mutação."""
        # Taxa de mutação alta para garantir que ocorra
        individual = Individual(self.sample_gene.copy(), self.distance_matrix)
        original_gene = individual.gene.copy()

        # Aplica mutação com alta probabilidade
        individual.mutate(swap_rate=1.0, two_opt_rate=1.0)

        # Verificar se o gene foi alterado ou aplicado two-opt
        # (Nota: two-opt pode não alterar o gene se já estiver otimizado)
        original_fitness = Individual(
            original_gene, self.distance_matrix).fitness
        self.assertTrue(
            not np.array_equal(individual.gene, original_gene) or
            individual.fitness <= original_fitness
        )

        # Verificar se todas as cidades ainda estão presentes
        self.assertEqual(set(individual.gene), set(range(1, self.n + 1)))

    def test_two_opt(self):
        """Testa a otimização local 2-opt."""
        individual = Individual(self.sample_gene.copy(), self.distance_matrix)

        # Definir um caminho não ótimo
        individual.gene = np.array([1, 2, 3, 4, 5])
        original_fitness = individual.fitness

        # Aplicar 2-opt (apply_two_opt em vez de apply_2opt)
        individual.apply_two_opt()

        # Verificar se o fitness melhorou ou permaneceu igual
        self.assertTrue(individual.fitness <= original_fitness)

        # Verificar se todas as cidades ainda estão presentes
        self.assertEqual(set(individual.gene), set(range(1, self.n + 1)))

    def test_genetic_optimizer(self):
        """Testa o otimizador genético."""
        optimizer = GeneticOptimizer(
            population_size=20,
            elite_size=0.2,
            mutation_rate=0.01,
            two_opt_rate=0.1
        )

        # Inicializa população
        population = optimizer.initialize_population(
            self.n, self.distance_matrix)

        # Verifica se a população inicial foi criada corretamente
        self.assertEqual(len(population), 20)

        # Executa algumas gerações
        final_population, generations = optimizer.evolve(
            population, generations=10)

        # Verifica se a população final é válida
        self.assertEqual(len(final_population), 20)

        # Verifica se os indivíduos têm genes válidos
        best_individual = final_population[0]
        self.assertEqual(len(best_individual.gene), self.n)
        self.assertEqual(set(best_individual.gene), set(range(1, self.n + 1)))

        # Verifica se o fitness é positivo
        self.assertTrue(best_individual.fitness > 0)


if __name__ == '__main__':
    unittest.main()
