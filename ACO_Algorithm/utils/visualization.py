"""
Módulo com funções para visualização de soluções do TSP e análise de algoritmos.
"""
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import List, Tuple, Optional, Dict, Any
from IPython.display import clear_output


def plot_tsp_solution(x: List[float], y: List[float], route: List[int], 
                      title: str = "Solução TSP", cost: Optional[float] = None,
                      save_path: Optional[str] = None) -> None:
    """
    Plota a solução do TSP.
    
    Args:
        x: Coordenadas x das cidades
        y: Coordenadas y das cidades
        route: Rota da solução (sequência de índices das cidades)
        title: Título do gráfico
        cost: Custo da solução (distância total)
        save_path: Caminho para salvar o gráfico. Se None, o gráfico não é salvo.
    """
    plt.figure(figsize=(10, 6))
    
    # Plota as cidades
    plt.scatter(x, y, c='blue', s=40)
    
    # Plota a rota
    for i in range(len(route) - 1):
        plt.plot([x[route[i]], x[route[i+1]]], [y[route[i]], y[route[i+1]]], 'k-', alpha=0.7)
    
    # Adiciona rótulos para as cidades
    for i in range(len(x)):
        plt.annotate(str(i), (x[i], y[i]), xytext=(5, 5), textcoords='offset points')
    
    # Adiciona o custo, se fornecido
    if cost is not None:
        title = f"{title} (Custo: {cost:.2f})"
    
    plt.title(title)
    plt.xlabel("Coordenada X")
    plt.ylabel("Coordenada Y")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Salva o gráfico, se solicitado
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.tight_layout()
    plt.show()


def plot_convergence(iterations: List[int], costs: List[float], 
                     title: str = "Convergência do Algoritmo", 
                     algorithm: str = "ACO",
                     save_path: Optional[str] = None) -> None:
    """
    Plota o gráfico de convergência do algoritmo.
    
    Args:
        iterations: Lista de iterações
        costs: Lista de custos correspondentes às iterações
        title: Título do gráfico
        algorithm: Nome do algoritmo
        save_path: Caminho para salvar o gráfico. Se None, o gráfico não é salvo.
    """
    plt.figure(figsize=(10, 6))
    
    plt.plot(iterations, costs, 'b-', linewidth=2)
    plt.title(title)
    plt.xlabel("Iteração" if algorithm == "ACO" else "Geração")
    plt.ylabel("Custo (Distância)")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Destacar o ponto de melhor solução
    best_idx = np.argmin(costs)
    best_iteration = iterations[best_idx]
    best_cost = costs[best_idx]
    
    plt.scatter(best_iteration, best_cost, c='red', s=100, zorder=5)
    plt.annotate(f"Melhor: {best_cost:.2f}", 
                 (best_iteration, best_cost),
                 xytext=(10, -20),
                 textcoords='offset points',
                 arrowprops=dict(arrowstyle='->', color='red'))
    
    # Salva o gráfico, se solicitado
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.tight_layout()
    plt.show()


class InteractiveVisualization:
    """Classe para visualizações interativas durante a execução dos algoritmos."""
    
    def __init__(self, x: List[float], y: List[float], title: str = "Evolução da Solução TSP",
                 update_interval: int = 5, algorithm: str = "ACO"):
        """
        Inicializa a visualização interativa.
        
        Args:
            x: Coordenadas x das cidades
            y: Coordenadas y das cidades
            title: Título da visualização
            update_interval: Intervalo de atualização (em iterações ou gerações)
            algorithm: Nome do algoritmo ("ACO" ou "GA")
        """
        self.x = x
        self.y = y
        self.title = title
        self.update_interval = update_interval
        self.algorithm = algorithm
        
        # Variáveis para acompanhar o progresso
        self.iterations = []
        self.costs = []
        self.best_route = None
        self.best_cost = float('inf')
        self.current_iteration = 0
        
        # Configuração da visualização
        plt.ion()  # Modo interativo
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(16, 6))
        self.fig.suptitle(title)
        
        # Configuração do gráfico de rota
        self.ax1.set_title("Melhor Rota Atual")
        self.ax1.set_xlabel("Coordenada X")
        self.ax1.set_ylabel("Coordenada Y")
        self.ax1.grid(True, linestyle='--', alpha=0.7)
        self.city_scatter = self.ax1.scatter(x, y, c='blue', s=40)
        self.route_line, = self.ax1.plot([], [], 'k-', alpha=0.7)
        
        # Configuração do gráfico de convergência
        self.ax2.set_title("Convergência do Algoritmo")
        self.ax2.set_xlabel("Iteração" if algorithm == "ACO" else "Geração")
        self.ax2.set_ylabel("Custo (Distância)")
        self.ax2.grid(True, linestyle='--', alpha=0.7)
        self.conv_line, = self.ax2.plot([], [], 'b-', linewidth=2)
        
        # Status do algoritmo
        self.status_text = self.fig.text(0.02, 0.02, "", fontsize=10)
        
        self.fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.pause(0.1)  # Pequena pausa para inicializar o modo interativo
    
    def update(self, route: List[int], cost: float, iteration: int) -> None:
        """
        Atualiza a visualização com a solução atual.
        
        Args:
            route: Rota atual
            cost: Custo da rota
            iteration: Iteração ou geração atual
        """
        self.current_iteration = iteration
        
        # Atualiza os dados de acompanhamento
        self.iterations.append(iteration)
        self.costs.append(cost)
        
        # Atualiza a melhor solução, se necessário
        if cost < self.best_cost:
            self.best_cost = cost
            self.best_route = route.copy()
        
        # Decide se deve atualizar a visualização
        if iteration % self.update_interval == 0 or iteration == 1:
            self._update_plots()
    
    def _update_plots(self) -> None:
        """Atualiza os gráficos da visualização."""
        # Limpa os gráficos
        self.ax1.cla()
        
        # Redesenha o gráfico da rota
        self.ax1.set_title("Melhor Rota Atual")
        self.ax1.scatter(self.x, self.y, c='blue', s=40)
        
        # Plota a melhor rota
        if self.best_route is not None:
            for i in range(len(self.best_route) - 1):
                self.ax1.plot(
                    [self.x[self.best_route[i]], self.x[self.best_route[i+1]]],
                    [self.y[self.best_route[i]], self.y[self.best_route[i+1]]],
                    'k-', alpha=0.7
                )
        
        # Adiciona rótulos para as cidades
        for i in range(len(self.x)):
            self.ax1.annotate(str(i), (self.x[i], self.y[i]), 
                             xytext=(5, 5), textcoords='offset points')
        
        self.ax1.set_xlabel("Coordenada X")
        self.ax1.set_ylabel("Coordenada Y")
        self.ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Atualiza o gráfico de convergência
        self.ax2.cla()
        self.ax2.set_title("Convergência do Algoritmo")
        self.ax2.plot(self.iterations, self.costs, 'b-', linewidth=2)
        
        # Destaca o ponto de melhor solução
        best_idx = np.argmin(self.costs)
        best_iteration = self.iterations[best_idx]
        best_cost = self.costs[best_idx]
        
        self.ax2.scatter(best_iteration, best_cost, c='red', s=100, zorder=5)
        self.ax2.annotate(f"Melhor: {best_cost:.2f}", 
                         (best_iteration, best_cost),
                         xytext=(10, -20),
                         textcoords='offset points',
                         arrowprops=dict(arrowstyle='->', color='red'))
        
        self.ax2.set_xlabel("Iteração" if self.algorithm == "ACO" else "Geração")
        self.ax2.set_ylabel("Custo (Distância)")
        self.ax2.grid(True, linestyle='--', alpha=0.7)
        
        # Atualiza o status
        iteration_name = "Iteração" if self.algorithm == "ACO" else "Geração"
        status = f"{iteration_name} atual: {self.current_iteration} | Melhor custo: {self.best_cost:.2f}"
        self.status_text.set_text(status)
        
        # Atualiza a visualização
        self.fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.pause(0.1)
    
    def finalize(self, save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Finaliza a visualização e retorna os dados finais.
        
        Args:
            save_path: Caminho para salvar a visualização final. Se None, não salva.
            
        Returns:
            Dict: Dicionário com os dados finais (melhor rota, custo, etc.)
        """
        # Atualizações finais
        self._update_plots()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            self.fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.ioff()  # Desativa modo interativo
        plt.show()  # Mostra o gráfico final
        
        return {
            "best_route": self.best_route,
            "best_cost": self.best_cost,
            "iterations": self.iterations,
            "costs": self.costs,
            "final_iteration": self.current_iteration
        }


def create_animation(x: List[float], y: List[float], routes: List[List[int]], 
                     costs: List[float], output_path: str = "animation.gif",
                     interval: int = 200) -> None:
    """
    Cria uma animação da evolução da solução.
    
    Args:
        x: Coordenadas x das cidades
        y: Coordenadas y das cidades
        routes: Lista de rotas ao longo das iterações
        costs: Lista de custos correspondentes às rotas
        output_path: Caminho para salvar a animação
        interval: Intervalo entre frames em milissegundos
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Criar o gráfico inicial
    city_scatter = ax.scatter(x, y, c='blue', s=40)
    route_line, = ax.plot([], [], 'k-', alpha=0.7)
    cost_text = ax.text(0.02, 0.95, "", transform=ax.transAxes)
    
    # Adiciona rótulos para as cidades
    for i in range(len(x)):
        ax.annotate(str(i), (x[i], y[i]), xytext=(5, 5), textcoords='offset points')
    
    ax.set_title("Evolução da Solução TSP")
    ax.grid(True, linestyle='--', alpha=0.7)
    
    def init():
        """Inicialização da animação."""
        route_line.set_data([], [])
        cost_text.set_text("")
        return route_line, cost_text
    
    def update(frame):
        """Atualização de cada frame da animação."""
        route = routes[frame]
        cost = costs[frame]
        
        # Extrai as coordenadas da rota
        route_x = [x[city] for city in route]
        route_y = [y[city] for city in route]
        
        # Fecha o ciclo
        if route[0] != route[-1]:
            route_x.append(route_x[0])
            route_y.append(route_y[0])
        
        # Atualiza a linha da rota
        route_line.set_data(route_x, route_y)
        
        # Atualiza o texto de custo
        cost_text.set_text(f"Iteração: {frame+1} | Custo: {cost:.2f}")
        
        return route_line, cost_text
    
    # Cria a animação
    anim = FuncAnimation(fig, update, frames=len(routes), init_func=init,
                         blit=True, interval=interval)
    
    # Salva a animação
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    anim.save(output_path, writer='pillow', fps=1000//interval)
    
    plt.close(fig)  # Fecha a figura para não exibi-la
