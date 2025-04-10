"""
Programa principal para resolver o Problema do Caixeiro Viajante usando Algoritmo Genético.
Inclui otimização bayesiana para encontrar os melhores hiperparâmetros.
"""
import time
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from genetic.individual import Individual
from genetic.optimizer import GeneticOptimizer
from genetic.utils import read_file, calculate_distance_matrix
from genetic.visualization import plot_route, plot_convergence, InteractiveGAVisualization, create_ga_animation
from bayesian_optimizer import BayesianOptimizer
from config import GeneticConfig


def parse_arguments():
    """
    Analisa os argumentos da linha de comando.
    
    Returns:
        argparse.Namespace: Argumentos analisados
    """
    parser = argparse.ArgumentParser(description='Algoritmo Genético para o TSP')
    parser.add_argument('--modo', type=str, default='padrao', choices=['padrao', 'bayesiano', 'config'],
                        help='Modo de execução: padrao, bayesiano ou config (usando arquivo de configuração)')
    parser.add_argument('--arquivo', type=str, default='Qatar.txt',
                        help='Nome do arquivo de entrada (deve estar em EntradasTSP/)')
    parser.add_argument('--geracoes', type=int, default=1000,
                        help='Número de gerações')
    parser.add_argument('--tam_populacao', type=int, default=100,
                        help='Tamanho da população')
    parser.add_argument('--tam_elite', type=float, default=0.1,
                        help='Proporção de elite (0-1)')
    parser.add_argument('--taxa_mutacao', type=float, default=0.01,
                        help='Taxa de mutação (0-1)')
    parser.add_argument('--taxa_crossover', type=float, default=0.8,
                        help='Taxa de crossover (0-1)')
    parser.add_argument('--use_2opt', action='store_true',
                        help='Aplicar otimização 2-opt')
    parser.add_argument('--config_file', type=str, default=None,
                        help='Caminho para arquivo de configuração (modo config)')
    parser.add_argument('--interactive', action='store_true',
                        help='Usar visualização interativa durante a execução')
    parser.add_argument('--animation', action='store_true',
                        help='Criar animação da evolução da solução')
    
    return parser.parse_args()


def run_standard_ga(arquivo, num_geracoes=1000, tam_populacao=100, 
                   tam_elite=0.1, taxa_mutacao=0.01, taxa_crossover=0.8,
                   use_2opt=True, interactive=False, animation=False):
    """
    Executa o algoritmo genético com parâmetros padrão.
    
    Args:
        arquivo: Caminho para o arquivo de entrada
        num_geracoes: Número de gerações para executar
        tam_populacao: Tamanho da população
        tam_elite: Proporção de elite (0-1)
        taxa_mutacao: Taxa de mutação (0-1)
        taxa_crossover: Taxa de crossover (0-1)
        use_2opt: Se deve aplicar otimização 2-opt
        interactive: Se deve usar visualização interativa
        animation: Se deve criar animação da evolução
        
    Returns:
        tuple: Melhor rota, melhor distância e dados de convergência
    """
    print(f"Resolvendo TSP para o arquivo: {os.path.basename(arquivo)}")
    
    # Iniciar temporizador
    inicio = time.time()
    
    # Ler dados do arquivo
    n, x, y = read_file(arquivo)
    
    # Calcular matriz de distâncias
    matriz_distancias = calculate_distance_matrix(n, x, y)
    
    # Visualização interativa
    if interactive:
        visualizer = InteractiveGAVisualization(
            x, y, 
            title=f"Evolução da Solução TSP - Algoritmo Genético",
            update_interval=10  # Atualiza a cada 10 gerações
        )
        
        # Configurar o otimizador genético
        otimizador = GeneticOptimizer(
            population_size=tam_populacao,
            elite_size=tam_elite,
            mutation_rate=taxa_mutacao,
            crossover_rate=taxa_crossover,
            generations=num_geracoes,
            distance_matrix=matriz_distancias,
            n_cities=n,
            use_2opt=use_2opt
        )
        
        # Lista para armazenar rotas e distâncias para possível animação
        all_routes = []
        all_distances = []
        
        # População inicial
        otimizador.create_initial_population()
        
        # Melhor rota inicial
        best_individual = otimizador.get_best_individual()
        best_route = best_individual.chromosome
        best_distance = 1 / best_individual.fitness
        
        # Atualiza a visualização com a população inicial
        visualizer.update(best_route, best_distance, 0)
        
        all_routes.append(best_route.copy())
        all_distances.append(best_distance)
        
        # Evolução da população
        for geracao in range(1, num_geracoes + 1):
            # Realiza uma geração
            otimizador.next_generation()
            
            # Obtém o melhor indivíduo atual
            best_individual = otimizador.get_best_individual()
            best_route = best_individual.chromosome
            best_distance = 1 / best_individual.fitness
            
            # Atualiza a visualização
            visualizer.update(best_route, best_distance, geracao)
            
            # Armazena para possível animação
            all_routes.append(best_route.copy())
            all_distances.append(best_distance)
            
            # Imprime progresso a cada 100 gerações
            if geracao % 100 == 0:
                print(f"Geração {geracao}/{num_geracoes}, Melhor distância: {best_distance:.2f}")
        
        # Finaliza a visualização
        vis_data = visualizer.finalize(
            save_path=f"resultados/ga/{os.path.basename(arquivo).split('.')[0]}_viz.png"
        )
        
        # Cria animação se solicitado
        if animation:
            output_path = f"resultados/ga/{os.path.basename(arquivo).split('.')[0]}_animation.gif"
            create_ga_animation(x, y, all_routes, all_distances, output_path)
        
        best_route = vis_data["best_route"]
        best_distance = vis_data["best_distance"]
        convergencia = {"gerações": vis_data["generations"], "distâncias": vis_data["distances"]}
        
    else:
        # Configurar o otimizador genético
        otimizador = GeneticOptimizer(
            population_size=tam_populacao,
            elite_size=tam_elite,
            mutation_rate=taxa_mutacao,
            crossover_rate=taxa_crossover,
            generations=num_geracoes,
            distance_matrix=matriz_distancias,
            n_cities=n,
            use_2opt=use_2opt
        )
        
        # Executar o algoritmo
        best_route, best_distance, convergencia = otimizador.evolve()
    
    # Calcular tempo de execução
    tempo_execucao = time.time() - inicio
    
    # Resultados
    print(f"Melhor distância encontrada: {best_distance:.2f}")
    print(f"Tempo de execução: {tempo_execucao:.2f} segundos")
    
    # Plotar a melhor rota
    plot_route(
        x, y, best_route, 
        title=f"Melhor Rota para {os.path.basename(arquivo)}",
        distance=best_distance,
        save_path=f"resultados/ga/{os.path.basename(arquivo).split('.')[0]}_rota.png"
    )
    
    # Plotar convergência
    plot_convergence(
        convergencia["gerações"], 
        convergencia["distâncias"],
        title=f"Convergência do AG para {os.path.basename(arquivo)}",
        save_path=f"resultados/ga/{os.path.basename(arquivo).split('.')[0]}_convergencia.png"
    )
    
    return best_route, best_distance, convergencia


def run_with_config(config=None):
    """
    Executa o algoritmo genético usando configurações de um arquivo YAML.
    
    Args:
        config: Instância de GeneticConfig. Se None, será criada uma nova.
        
    Returns:
        tuple: Melhor rota, melhor distância e dados de convergência
    """
    if config is None:
        config = GeneticConfig()
    
    print(f"Executando GA com configuração do arquivo: {config.config_file}")
    
    # Garantir que o diretório de resultados existe
    os.makedirs("resultados/ga", exist_ok=True)
    
    return run_standard_ga(
        arquivo=config.input_file,
        num_geracoes=config.num_generations,
        tam_populacao=config.population_size,
        tam_elite=config.elite_size,
        taxa_mutacao=config.mutation_rate,
        taxa_crossover=config.crossover_rate,
        use_2opt=config.use_two_opt,
        interactive=config.use_interactive_viz,
        animation=config.get('visualization', 'save_animation', False)
    )


def main():
    """
    Função principal do programa.
    """
    args = parse_arguments()
    
    # Garantir que o diretório de resultados existe
    os.makedirs("resultados/ga", exist_ok=True)
    
    if args.modo == 'config':
        # Executar usando arquivo de configuração
        config = GeneticConfig(args.config_file)
        best_route, best_distance, _ = run_with_config(config)
    
    elif args.modo == 'padrao':
        # Caminho para o arquivo de entrada
        arquivo = os.path.join('EntradasTSP', args.arquivo)
        if not os.path.exists(arquivo):
            print(f"Arquivo não encontrado: {arquivo}")
            return
        
        best_route, best_distance, _ = run_standard_ga(
            arquivo=arquivo,
            num_geracoes=args.geracoes,
            tam_populacao=args.tam_populacao,
            tam_elite=args.tam_elite,
            taxa_mutacao=args.taxa_mutacao,
            taxa_crossover=args.taxa_crossover,
            use_2opt=args.use_2opt,
            interactive=args.interactive,
            animation=args.animation
        )
    
    elif args.modo == 'bayesiano':
        # Caminho para o arquivo de entrada
        arquivo = os.path.join('EntradasTSP', args.arquivo)
        if not os.path.exists(arquivo):
            print(f"Arquivo não encontrado: {arquivo}")
            return
        
        # Executar otimização bayesiana
        bo = BayesianOptimizer(arquivo, n_calls=20, n_generations=100)
        best_params, best_fitness = bo.optimize()
        
        print("\nMelhores parâmetros encontrados:")
        print(f"Tamanho da população: {best_params['population_size']}")
        print(f"Tamanho da elite: {best_params['elite_size']}")
        print(f"Taxa de mutação: {best_params['mutation_rate']}")
        print(f"Melhor fitness: {best_fitness}")
        
        # Executar com os melhores parâmetros
        best_route, best_distance, _ = run_standard_ga(
            arquivo=arquivo,
            num_geracoes=args.geracoes,
            tam_populacao=best_params['population_size'],
            tam_elite=best_params['elite_size'],
            taxa_mutacao=best_params['mutation_rate'],
            taxa_crossover=args.taxa_crossover,
            use_2opt=args.use_2opt,
            interactive=args.interactive,
            animation=args.animation
        )
    
    else:
        print(f"Modo desconhecido: {args.modo}")
        return


if __name__ == "__main__":
    main()
