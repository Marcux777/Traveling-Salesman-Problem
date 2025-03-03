import math
from Max_Min_Ant_System import MMAS
from Ant_Colony_Optimization import ACO
import time
import numpy as np
from multiprocessing import Pool, cpu_count


def calculate_Weights(x, y):
    x = np.array(x)
    y = np.array(y)

    dx = x[:, np.newaxis] - x
    dy = y[:, np.newaxis] - y

    distances = np.sqrt(dx**2 + dy**2)

    np.fill_diagonal(distances, np.inf)

    return distances


def read_archive(caminho_arquivo):
    x, y = [], []
    with open(caminho_arquivo, "r") as arquivo:
        n = int(next(arquivo).strip())
        x = [float("inf")] * n
        y = [float("inf")] * n
        for linha in arquivo:
            l, c1, c2 = map(float, linha.split())
            l = int(l) - 1
            x[l], y[l] = c1, c2
    return n, x, y


def run_aco(graph, num_ants, alpha, beta, rho, start_city, max_iterations):
    aco = ACO(graph, num_ants=num_ants, alpha=alpha, beta=beta, rho=rho)
    best_solution, best_cost = aco.run(start_city, max_iterations)
    return best_solution, best_cost


if __name__ == "__main__":
    pasta = "EntradasTSP/"
    lista_arquivos = [
        "Djibouti.txt",
        "Qatar.txt",
        "Argentina.txt",
        "Burma.txt",
        "China.txt",
        "Egypt.txt",
        "Finland.txt",
        "Greece.txt",
        "Honduras.txt",
        "Luxembourg.txt",
        "Zimbabwe.txt",
        "Uruguay.txt",
        "Yemen.txt",
        "Western Sahara.txt",
        "Vietnam.txt",
        "Tanzania.txt",
        "Sweden.txt",
        "Rwanda.txt",
        "Ireland.txt",
        "Japan.txt",
        "Kazakhstan.txt",
        "Morocco.txt",
        "Nicaragua.txt",
        "Oman.txt",
        "Panama.txt",
    ]

    n, x, y = read_archive(pasta + lista_arquivos[0])
    graph = calculate_Weights(x, y)

    num_ants = 30
    alpha = 0.8
    beta = 0.8
    rho = 0.9
    start_city = 0
    max_iterations = 100

    start_time = time.time()

    # Paralelização usando multiprocessing
    with Pool(cpu_count()) as pool:
        results = pool.starmap(
            run_aco,
            [
                (graph, num_ants, alpha, beta, rho, start_city, max_iterations)
                for _ in range(cpu_count())
            ],
        )

    best_solution, best_cost = min(results, key=lambda x: x[1])

    end_time = time.time()

    print(*best_solution)
    print(best_cost)
    execution_time = end_time - start_time
    print("Execution time:", execution_time, "seconds")
