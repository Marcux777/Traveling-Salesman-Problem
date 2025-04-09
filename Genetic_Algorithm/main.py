"""
Programa principal para resolver o Problema do Caixeiro Viajante usando Algoritmo Genético.
Inclui otimização bayesiana para encontrar os melhores hiperparâmetros.
"""

import time
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from genetic import Individual, GeneticOptimizer, read_file, calculate_distance_matrix, plot_convergence, plot_route
from bayesian_optimizer import BayesianOptimizer


def run_standard_ga(arquivo, num_geracoes=1000):
    """
    Executa o algoritmo genético com parâmetros padrão.

    Args:
        arquivo: Caminho para o arquivo de entrada
        num_geracoes: Número de gerações para executar
    """
    print(f"Resolvendo TSP para o arquivo: {os.path.basename(arquivo)}")

    # Iniciar temporizador
    inicio = time.time()

    # Ler dados do arquivo
    n, x, y = read_file(arquivo)

    # Calcular matriz de distâncias
    matriz_distancias = calculate_distance_matrix(n, x, y)

    # Configurar o otimizador genético
    otimizador = GeneticOptimizer(
        population_size=100,
        elite_size=0.1,
        mutation_rate=0.01,
        two_opt_rate=0.1,
        crossover_rate=0.7,
        use_parallel=True
    )

    # Inicializar população
    populacao = [Individual(Individual.create_random_path(n), matriz_distancias)
                 for _ in range(otimizador.population_size)]

    # Evoluir população
    resultados, geracao_final = otimizador.evolve(populacao, num_geracoes)

    # Exibir resultados
    melhor_individuo = resultados[0]
    print(f"Geração: {geracao_final}")
    print(f"Melhor caminho: {melhor_individuo.gene}")
    print(f"Custo: {round(melhor_individuo.fitness, 2)}")

    # Finalizar temporizador e imprimir tempo de execução
    fim = time.time()
    tempo_execucao = fim - inicio
    print(f"Tempo de execução: {tempo_execucao:.2f} segundos")

    # Plotar gráfico de convergência
    plot_convergence(otimizador.best_fitnesses)

    # Plotar a melhor rota encontrada
    plot_route(melhor_individuo, x, y)

    return resultados, otimizador


def run_bayesian_optimization(arquivo, n_calls=20, n_random_starts=5, n_generations=200):
    """
    Executa otimização bayesiana para encontrar os melhores hiperparâmetros.

    Args:
        arquivo: Caminho para o arquivo de entrada
        n_calls: Número de chamadas para a otimização bayesiana
        n_random_starts: Número de pontos aleatórios iniciais
        n_generations: Número de gerações para cada execução do GA
    """
    # Criar e configurar o otimizador bayesiano
    bayesian_opt = BayesianOptimizer(
        input_file=arquivo,
        n_calls=n_calls,
        n_random_starts=n_random_starts,
        n_generations=n_generations,
        verbose=True
    )

    # Executar a otimização
    best_params = bayesian_opt.optimize()

    # Plotar resultados da otimização
    bayesian_opt.plot_results()

    # Executar o GA com os melhores parâmetros
    resultados, tempo_execucao = bayesian_opt.run_with_optimal_params()

    # Retornar os resultados e os melhores parâmetros
    return resultados, best_params, bayesian_opt


def main():
    """
    Função principal que executa o algoritmo genético para resolver o TSP.
    """
    # Configurar o parser de argumentos
    parser = argparse.ArgumentParser(
        description='Resolver o TSP usando Algoritmo Genético com otimização bayesiana de hiperparâmetros.')
    parser.add_argument('--modo', type=str, choices=['padrao', 'bayesiano'], default='padrao',
                        help='Modo de execução: padrao ou bayesiano (default: padrao)')
    parser.add_argument('--arquivo', type=str, default='Qatar.txt',
                        help='Nome do arquivo de entrada (default: Qatar.txt)')
    parser.add_argument('--geracoes', type=int, default=1000,
                        help='Número de gerações para o GA (default: 1000)')
    parser.add_argument('--chamadas', type=int, default=20,
                        help='Número de chamadas para otimização bayesiana (default: 20)')
    parser.add_argument('--inicios', type=int, default=5,
                        help='Número de pontos aleatórios iniciais para otimização bayesiana (default: 5)')
    parser.add_argument('--geracoes_bo', type=int, default=200,
                        help='Número de gerações para cada chamada na otimização bayesiana (default: 200)')

    args = parser.parse_args()

    # Lista de arquivos disponíveis
    lista_arquivos = [
        'Djibouti.txt', 'Qatar.txt', 'Argentina.txt', 'Burma.txt', 'China.txt',
        'Egypt.txt', 'Finland.txt', 'Greece.txt', 'Honduras.txt', 'Luxembourg.txt',
    ]

    # Pasta com os arquivos de entrada
    pasta = "/workspaces/Traveling-Salesman-Problem/EntradasTSP/"

    # Verificar se o arquivo existe
    if args.arquivo not in lista_arquivos:
        print(
            f"Arquivo {args.arquivo} não encontrado. Usando Qatar.txt como padrão.")
        args.arquivo = 'Qatar.txt'

    # Caminho completo para o arquivo
    arquivo_completo = os.path.join(pasta, args.arquivo)

    # Executar no modo apropriado
    if args.modo == 'padrao':
        print(f"Executando algoritmo genético no modo padrão")
        run_standard_ga(arquivo_completo, args.geracoes)
    else:
        print(f"Executando algoritmo genético com otimização bayesiana")
        run_bayesian_optimization(
            arquivo_completo,
            n_calls=args.chamadas,
            n_random_starts=args.inicios,
            n_generations=args.geracoes_bo
        )


if __name__ == "__main__":
    main()
