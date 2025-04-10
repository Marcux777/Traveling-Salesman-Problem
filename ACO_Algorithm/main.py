"""
Módulo principal para executar o Algoritmo de Colônia de Formigas para o Problema do Caixeiro Viajante.
"""

import argparse
import os

from ACO_Algorithm.utils.tsp_utils import calculate_distances, read_tsp_file, find_tsp_file_path
from ACO_Algorithm.aco.runner import run_standard_aco
from ACO_Algorithm.optimization.bayesian_optimizer import run_bayesian_optimization


def parse_arguments():
    """
    Configura e analisa os argumentos da linha de comando.

    Returns:
        argparse.Namespace: Objeto contendo os argumentos da linha de comando.
    """
    parser = argparse.ArgumentParser(
        description='Resolver o TSP usando ACO (Ant Colony Optimization) com otimização bayesiana de hiperparâmetros.')

    parser.add_argument('--modo', type=str, choices=['padrao', 'bayesiano'], default='padrao',
                        help='Modo de execução: padrao ou bayesiano (default: padrao)')
    parser.add_argument('--arquivo', type=str, default='Djibouti.txt',
                        help='Nome do arquivo de entrada (default: Djibouti.txt)')
    parser.add_argument('--num_formigas', type=int, default=30,
                        help='Número de formigas para o modo padrão (default: 30)')
    parser.add_argument('--alpha', type=float, default=0.8,
                        help='Parâmetro alpha para o modo padrão (default: 0.8)')
    parser.add_argument('--beta', type=float, default=0.8,
                        help='Parâmetro beta para o modo padrão (default: 0.8)')
    parser.add_argument('--rho', type=float, default=0.9,
                        help='Parâmetro rho para o modo padrão (default: 0.9)')
    parser.add_argument('--iteracoes', type=int, default=100,
                        help='Número de iterações para o modo padrão (default: 100)')
    parser.add_argument('--chamadas', type=int, default=20,
                        help='Número de chamadas para otimização bayesiana (default: 20)')
    parser.add_argument('--inicios', type=int, default=5,
                        help='Número de pontos aleatórios iniciais para otimização bayesiana (default: 5)')
    parser.add_argument('--iteracoes_bo', type=int, default=50,
                        help='Número de iterações para cada chamada na otimização bayesiana (default: 50)')
    parser.add_argument('--k_opt', type=int, default=2,
                        help='Valor de k para K-Opt (default: 2)')

    return parser.parse_args()


def main():
    """
    Função principal para execução do algoritmo ACO.
    """
    # Obter argumentos da linha de comando
    args = parse_arguments()

    # Lista de arquivos disponíveis
    lista_arquivos = [
        "Djibouti.txt", "Qatar.txt", "Argentina.txt", "Burma.txt", "China.txt",
        "Egypt.txt", "Finland.txt", "Greece.txt", "Honduras.txt", "Luxembourg.txt",
        "Zimbabwe.txt", "Uruguay.txt", "Yemen.txt", "Western Sahara.txt", "Vietnam.txt",
        "Tanzania.txt", "Sweden.txt", "Rwanda.txt", "Ireland.txt", "Japan.txt",
        "Kazakhstan.txt", "Morocco.txt", "Nicaragua.txt", "Oman.txt", "Panama.txt",
    ]

    # Verificar se o arquivo existe e obter o caminho
    if args.arquivo not in lista_arquivos:
        print(
            f"Arquivo {args.arquivo} não encontrado. Usando Djibouti.txt como padrão.")
        args.arquivo = 'Djibouti.txt'

    # Encontrar o caminho do arquivo
    arquivo_completo = find_tsp_file_path(args.arquivo)

    # Ler dados do arquivo
    n, x, y = read_tsp_file(arquivo_completo)

    # Calcular matriz de distâncias
    graph = calculate_distances(x, y)

    # Executar o modo apropriado
    if args.modo == 'padrao':
        print(f"Executando ACO no modo padrão")
        run_standard_aco(
            graph,
            x,
            y,
            num_ants=args.num_formigas,
            alpha=args.alpha,
            beta=args.beta,
            rho=args.rho,
            max_iterations=args.iteracoes,
            k_value=args.k_opt
        )
    else:
        print(f"Executando ACO com otimização bayesiana")
        run_bayesian_optimization(
            graph,
            x,
            y,
            n_calls=args.chamadas,
            n_random_starts=args.inicios,
            max_iterations=args.iteracoes_bo,
            k_value=args.k_opt
        )


if __name__ == "__main__":
    main()
