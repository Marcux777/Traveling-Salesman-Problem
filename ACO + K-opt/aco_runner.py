"""
Módulo que implementa funções para execução paralela do Algoritmo de Colônia de Formigas (ACO).
"""

import time
from multiprocessing import Pool, cpu_count

from Ant_Colony_Optimization import ACO
from K_Opt import K_Opt
from utils import plot_tsp_solution


def run_aco_single(params):
    """
    Executa uma única instância do algoritmo ACO.
    
    Args:
        params (tuple): Uma tupla com os parâmetros necessários:
                       (graph, num_ants, alpha, beta, rho, start_city, max_iterations)
    
    Returns:
        tuple: Uma tupla contendo a melhor solução e seu custo.
    """
    graph, num_ants, alpha, beta, rho, start_city, max_iterations = params
    aco = ACO(graph, num_ants=num_ants, alpha=alpha, beta=beta, rho=rho)
    best_solution, best_cost = aco.run(start_city, max_iterations)
    return best_solution, best_cost


def run_parallel_aco(graph, num_ants, alpha, beta, rho, start_city, max_iterations, num_processes=None):
    """
    Executa múltiplas instâncias do algoritmo ACO em paralelo.
    
    Args:
        graph (numpy.ndarray): Matriz de distâncias entre as cidades.
        num_ants (int): Número de formigas.
        alpha (float): Importância do feromônio.
        beta (float): Importância da heurística.
        rho (float): Taxa de evaporação do feromônio.
        start_city (int): Cidade inicial.
        max_iterations (int): Número máximo de iterações.
        num_processes (int, optional): Número de processos paralelos.
                                      Se None, usa o número de CPUs disponíveis.
    
    Returns:
        tuple: Uma tupla contendo a melhor solução e seu custo.
    """
    if num_processes is None:
        num_processes = cpu_count()
    
    params = (graph, num_ants, alpha, beta, rho, start_city, max_iterations)
    params_list = [params] * num_processes
    
    with Pool(num_processes) as pool:
        results = pool.map(run_aco_single, params_list)
    
    return min(results, key=lambda x: x[1])


def run_standard_aco(graph, x, y, num_ants=30, alpha=0.8, beta=0.8, rho=0.9, 
                    start_city=0, max_iterations=100, apply_k_opt=True, k_value=2):
    """
    Executa o algoritmo ACO padrão com paralelização e mostra os resultados.
    
    Args:
        graph (numpy.ndarray): Matriz de distâncias entre as cidades.
        x (list): Lista de coordenadas x das cidades.
        y (list): Lista de coordenadas y das cidades.
        num_ants (int): Número de formigas.
        alpha (float): Importância do feromônio.
        beta (float): Importância da heurística.
        rho (float): Taxa de evaporação do feromônio.
        start_city (int): Cidade inicial.
        max_iterations (int): Número máximo de iterações.
        apply_k_opt (bool): Se deve aplicar a heurística K-Opt.
        k_value (int): Valor de k para K-Opt.
    
    Returns:
        tuple: Uma tupla contendo a melhor solução e seu custo.
    """
    print(f"Executando ACO padrão com {num_ants} formigas, alpha={alpha}, beta={beta}, rho={rho}")
    print(f"Número de cidades: {len(x)}")
    
    start_time = time.time()
    
    # Executa ACO em paralelo
    best_solution, best_cost = run_parallel_aco(
        graph, num_ants, alpha, beta, rho, start_city, max_iterations
    )
    
    # Aplica K-Opt na melhor solução
    if apply_k_opt:
        print(f"Aplicando {k_value}-opt na melhor solução...")
        k_opt = K_Opt(graph)
        improved_solution, improved_cost = k_opt.k_opt(best_solution, k_value)
        
        if improved_cost < best_cost:
            print(f"K-Opt melhorou a solução: {best_cost:.2f} -> {improved_cost:.2f}")
            best_solution = improved_solution
            best_cost = improved_cost
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    print("Melhor solução:", *best_solution)
    print("Custo:", best_cost)
    print("Tempo de execução:", execution_time, "segundos")
    
    # Plotar a solução
    plot_tsp_solution(best_solution, x, y, best_cost)
    
    return best_solution, best_cost