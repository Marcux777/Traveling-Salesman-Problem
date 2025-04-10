"""
Módulo para carregar e gerenciar configurações dos algoritmos.
"""
import os
import yaml
from typing import Dict, Any, Optional


class Config:
    """Classe base para configuração dos algoritmos."""

    def __init__(self, config_file: str):
        """
        Inicializa a configuração a partir de um arquivo YAML.
        
        Args:
            config_file: Caminho para o arquivo de configuração YAML
        """
        self.config_file = config_file
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Carrega a configuração do arquivo YAML.
        
        Returns:
            Dict: Dicionário com as configurações
        """
        try:
            with open(self.config_file, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            print(f"Erro ao carregar configuração: {e}")
            return {}
    
    def save_config(self) -> None:
        """Salva a configuração atual no arquivo YAML."""
        try:
            with open(self.config_file, 'w') as file:
                yaml.dump(self.config, file, default_flow_style=False)
        except Exception as e:
            print(f"Erro ao salvar configuração: {e}")
    
    def get(self, section: str, key: str, default: Any = None) -> Any:
        """
        Obtém um valor da configuração.
        
        Args:
            section: Seção da configuração
            key: Chave do valor
            default: Valor padrão se a chave não existir
            
        Returns:
            Any: Valor da configuração
        """
        try:
            return self.config.get(section, {}).get(key, default)
        except:
            return default
    
    def set(self, section: str, key: str, value: Any) -> None:
        """
        Define um valor na configuração.
        
        Args:
            section: Seção da configuração
            key: Chave do valor
            value: Valor a ser definido
        """
        if section not in self.config:
            self.config[section] = {}
        self.config[section][key] = value


class ACOConfig(Config):
    """Configuração específica para o algoritmo ACO."""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Inicializa a configuração ACO.
        
        Args:
            config_file: Caminho para o arquivo de configuração. Se None, usa o padrão.
        """
        if config_file is None:
            # Usa o caminho padrão relativo à raiz do projeto
            root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            config_file = os.path.join(root_dir, 'config', 'aco_config.yaml')
        super().__init__(config_file)
    
    @property
    def input_file(self) -> str:
        """Obtém o arquivo de entrada."""
        return self.get('general', 'input_file', 'EntradasTSP/Qatar.txt')
    
    @property
    def max_iterations(self) -> int:
        """Obtém o número máximo de iterações."""
        return self.get('general', 'max_iterations', 100)
    
    @property
    def num_ants(self) -> int:
        """Obtém o número de formigas."""
        return self.get('aco', 'num_ants', 30)
    
    @property
    def alpha(self) -> float:
        """Obtém o valor de alpha."""
        return self.get('aco', 'alpha', 1.0)
    
    @property
    def beta(self) -> float:
        """Obtém o valor de beta."""
        return self.get('aco', 'beta', 2.0)
    
    @property
    def rho(self) -> float:
        """Obtém o valor de rho."""
        return self.get('aco', 'rho', 0.5)
    
    @property
    def use_mmas(self) -> bool:
        """Verifica se deve usar MMAS."""
        return self.get('mmas', 'enabled', True)
    
    @property
    def p_best(self) -> float:
        """Obtém o valor de p_best para MMAS."""
        return self.get('mmas', 'p_best', 0.05)
    
    @property
    def stagnation_limit(self) -> int:
        """Obtém o limite de estagnação para MMAS."""
        return self.get('mmas', 'stagnation_limit', 10)
    
    @property
    def use_k_opt(self) -> bool:
        """Verifica se deve usar K-Opt."""
        return self.get('k_opt', 'enabled', True)
    
    @property
    def k_value(self) -> int:
        """Obtém o valor de K para K-Opt."""
        return self.get('k_opt', 'k_value', 2)
    
    @property
    def use_interactive_viz(self) -> bool:
        """Verifica se deve usar visualização interativa."""
        return self.get('visualization', 'interactive', True)
    
    @property
    def update_interval(self) -> int:
        """Obtém o intervalo de atualização para visualização interativa."""
        return self.get('visualization', 'update_interval', 5)


class GeneticConfig(Config):
    """Configuração específica para o algoritmo Genético."""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Inicializa a configuração Genética.
        
        Args:
            config_file: Caminho para o arquivo de configuração. Se None, usa o padrão.
        """
        if config_file is None:
            # Usa o caminho padrão relativo à raiz do projeto
            root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            config_file = os.path.join(root_dir, 'config', 'genetic_config.yaml')
        super().__init__(config_file)
    
    @property
    def input_file(self) -> str:
        """Obtém o arquivo de entrada."""
        return self.get('general', 'input_file', 'EntradasTSP/Qatar.txt')
    
    @property
    def num_generations(self) -> int:
        """Obtém o número de gerações."""
        return self.get('general', 'num_generations', 1000)
    
    @property
    def population_size(self) -> int:
        """Obtém o tamanho da população."""
        return self.get('genetic', 'population_size', 100)
    
    @property
    def elite_size(self) -> float:
        """Obtém a proporção de elite."""
        return self.get('genetic', 'elite_size', 0.1)
    
    @property
    def mutation_rate(self) -> float:
        """Obtém a taxa de mutação."""
        return self.get('genetic', 'mutation_rate', 0.01)
    
    @property
    def crossover_rate(self) -> float:
        """Obtém a taxa de crossover."""
        return self.get('genetic', 'crossover_rate', 0.8)
    
    @property
    def use_two_opt(self) -> bool:
        """Verifica se deve usar otimização 2-opt."""
        return self.get('genetic', 'use_two_opt', True)
    
    @property
    def use_interactive_viz(self) -> bool:
        """Verifica se deve usar visualização interativa."""
        return self.get('visualization', 'interactive', True)
    
    @property
    def update_interval(self) -> int:
        """Obtém o intervalo de atualização para visualização interativa."""
        return self.get('visualization', 'update_interval', 10)