"""
Módulo principal para executar o Algoritmo de Colônia de Formigas para o Problema do Caixeiro Viajante.
"""

import argparse
import os
import time
import numpy as np

from ACO_Algorithm.utils.tsp_utils import calculate_distances, read_tsp_file, find_tsp_file_path
from ACO_Algorithm.aco.colony import ACO
from ACO_Algorithm.aco.mmas import MMAS
from ACO_Algorithm.aco.runner import run_standard_aco
from ACO_Algorithm.optimization.bayesian_optimizer import run_bayesian_optimization
from ACO_Algorithm.utils.visualization import InteractiveVisualization, create_animation
from config import ACOConfig


def parse_arguments():
    """
    Analisa os argumentos da linha de comando.

    Returns:
        argparse.Namespace: Argumentos analisados
    """
    parser = argparse.ArgumentParser(
        description='Algoritmo de Colônia de Formigas para o TSP')
    parser.add_argument('--modo', type=str, default='padrao', choices=['padrao', 'bayesiano', 'config'],
                        help='Modo de execução: padrao, bayesiano ou config (usando arquivo de configuração)')
    parser.add_argument('--arquivo', type=str, default='Qatar.txt',
                        help='Nome do arquivo de entrada (deve estar em EntradasTSP/)')
    parser.add_argument('--num_formigas', type=int, default=30,
                        help='Número de formigas')
    parser.add_argument('--alpha', type=float, default=0.8,
                        help='Importância do feromônio (alpha)')
    parser.add_argument('--beta', type=float, default=0.8,
                        help='Importância da heurística (beta)')
    parser.add_argument('--rho', type=float, default=0.9,
                        help='Taxa de evaporação do feromônio (rho)')
    parser.add_argument('--iteracoes', type=int, default=100,
                        help='Número máximo de iterações')
    parser.add_argument('--chamadas', type=int, default=20,
                        help='Número de chamadas para otimização bayesiana')
    parser.add_argument('--inicios', type=int, default=5,
                        help='Número de pontos iniciais aleatórios para otimização bayesiana')
    parser.add_argument('--iteracoes_bo', type=int, default=50,
                        help='Número de iterações para cada execução durante otimização bayesiana')
    parser.add_argument('--k_opt', type=int, default=2,
                        help='Valor de k para K-Opt (2 = 2-opt, 3 = 3-opt, etc.)')
    parser.add_argument('--use_mmas', action='store_true',
                        help='Usar Max-Min Ant System em vez do ACO padrão')
    parser.add_argument('--config_file', type=str, default=None,
                        help='Caminho para arquivo de configuração (modo config)')
    parser.add_argument('--interactive', action='store_true',
                        help='Usar visualização interativa durante a execução')
    parser.add_argument('--animation', action='store_true',
                        help='Criar animação da evolução da solução')

    return parser.parse_args()


def run_with_config(config=None):
    """
    Executa o algoritmo ACO usando configurações de um arquivo YAML.

    Args:
        config: Instância de ACOConfig. Se None, será criada uma nova.

    Returns:
        tuple: Melhor solução e custo
    """
    if config is None:
        config = ACOConfig()

    print(f"Executando ACO com configuração do arquivo: {config.config_file}")

    # Ler arquivo de entrada
    input_filepath = find_tsp_file_path(config.input_file)
    if not input_filepath:
        print(f"Arquivo de entrada não encontrado: {config.input_file}")
        return None, None

    n, x, y = read_tsp_file(input_filepath)
    graph = calculate_distances(x, y)

    # Configuração do algoritmo
    start_city = 0

    # Decide se deve usar visualização interativa
    if config.use_interactive_viz:
        visualizer = InteractiveVisualization(
            x, y,
            title=f"Evolução da Solução TSP - {'MMAS' if config.use_mmas else 'ACO'}",
            update_interval=config.update_interval
        )

        # Inicializa o algoritmo apropriado
        if config.use_mmas:
            algorithm = MMAS(
                graph,
                num_ants=config.num_ants,
                alpha=config.alpha,
                beta=config.beta,
                rho=config.rho,
                p_best=config.p_best,
                stagnation_limit=config.stagnation_limit
            )
        else:
            algorithm = ACO(
                graph,
                num_ants=config.num_ants,
                alpha=config.alpha,
                beta=config.beta,
                rho=config.rho
            )

        # Lista para armazenar soluções e custos (para possível animação)
        all_routes = []
        all_costs = []

        # Executa o algoritmo com visualização interativa
        start_time = time.time()
        best_solution = None
        best_cost = float('inf')

        for iteration in range(1, config.max_iterations + 1):
            # Constrói soluções
            routes = algorithm.construct_solutions(start_city)
            costs = [algorithm.calculate_cost(route) for route in routes]

            # Encontra a melhor solução da iteração
            current_best_idx = np.argmin(costs)
            current_best_solution = routes[current_best_idx]
            current_best_cost = costs[current_best_idx]

            # Atualiza a melhor solução global se necessário
            if current_best_cost < best_cost:
                best_cost = current_best_cost
                best_solution = current_best_solution.copy()

            # Atualiza a visualização
            visualizer.update(best_solution, best_cost, iteration)

            # Atualiza a matriz de feromônios
            algorithm.update_pheromones(routes, costs)

            # Armazena para possível animação
            all_routes.append(best_solution.copy())
            all_costs.append(best_cost)

        # Finaliza a visualização
        visualizer.finalize(
            save_path=f"resultados/aco/{os.path.basename(config.input_file).split('.')[0]}_viz.png")

        # Cria animação se solicitado
        if config.get('visualization', 'save_animation', False):
            output_path = f"resultados/aco/{os.path.basename(config.input_file).split('.')[0]}_animation.gif"
            create_animation(x, y, all_routes, all_costs, output_path)

        elapsed_time = time.time() - start_time
        print(f"Tempo total: {elapsed_time:.2f} segundos")
        return best_solution, best_cost
    else:
        # Execução padrão sem visualização interativa
        return run_standard_aco(
            graph, x, y,
            num_ants=config.num_ants,
            alpha=config.alpha,
            beta=config.beta,
            rho=config.rho,
            start_city=start_city,
            max_iterations=config.max_iterations,
            apply_k_opt=config.use_k_opt,
            k_value=config.k_value,
            use_mmas=config.use_mmas
        )


def main():
    """
    Função principal para execução do algoritmo ACO.
    """
    args = parse_arguments()

    if args.modo == 'config':
        # Executar usando arquivo de configuração
        config = ACOConfig(args.config_file)
        best_solution, best_cost = run_with_config(config)

    elif args.modo == 'padrao':
        # Encontrar o caminho completo do arquivo
        input_filepath = find_tsp_file_path(args.arquivo)
        if not input_filepath:
            print(f"Arquivo de entrada não encontrado: {args.arquivo}")
            return

        # Ler dados do arquivo
        n, x, y = read_tsp_file(input_filepath)
        graph = calculate_distances(x, y)

        if args.interactive:
            # Cria configuração a partir dos argumentos da linha de comando
            config = ACOConfig()
            config.set('general', 'input_file', args.arquivo)
            config.set('general', 'max_iterations', args.iteracoes)
            config.set('aco', 'num_ants', args.num_formigas)
            config.set('aco', 'alpha', args.alpha)
            config.set('aco', 'beta', args.beta)
            config.set('aco', 'rho', args.rho)
            config.set('mmas', 'enabled', args.use_mmas)
            config.set('k_opt', 'enabled', args.k_opt > 0)
            config.set('k_opt', 'k_value', args.k_opt)
            config.set('visualization', 'interactive', True)
            config.set('visualization', 'save_animation', args.animation)

            best_solution, best_cost = run_with_config(config)
        else:
            # Executar ACO padrão
            best_solution, best_cost = run_standard_aco(
                graph, x, y,
                num_ants=args.num_formigas,
                alpha=args.alpha,
                beta=args.beta,
                rho=args.rho,
                max_iterations=args.iteracoes,
                k_value=args.k_opt,
                use_mmas=args.use_mmas
            )

    elif args.modo == 'bayesiano':
        # Encontrar o caminho completo do arquivo
        input_filepath = find_tsp_file_path(args.arquivo)
        if not input_filepath:
            print(f"Arquivo de entrada não encontrado: {args.arquivo}")
            return

        # Ler dados do arquivo
        n, x, y = read_tsp_file(input_filepath)
        graph = calculate_distances(x, y)

        # Executar otimização bayesiana
        print(
            f"Executando otimização bayesiana para o arquivo: {args.arquivo}")
        print(f"Chamadas: {args.chamadas}, Pontos iniciais: {args.inicios}")
        best_solution, best_cost = run_bayesian_optimization(
            graph, x, y,
            n_calls=args.chamadas,
            n_random_starts=args.inicios,
            max_iterations=args.iteracoes_bo,
            k_value=args.k_opt
        )

    else:
        print(f"Modo desconhecido: {args.modo}")
        return

    if best_solution is not None:
        print(f"Melhor custo encontrado: {best_cost:.2f}")


if __name__ == "__main__":
    main()
