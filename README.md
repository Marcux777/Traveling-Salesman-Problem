# Problema do Caixeiro Viajante (Traveling Salesman Problem)

Este repositório contém implementações de diferentes algoritmos para resolver o Problema do Caixeiro Viajante (TSP), incluindo:
- Algoritmo Genético (GA)
- Otimização por Colônia de Formigas (ACO)
- K-opt local search

## Estrutura do Projeto

O projeto está organizado nos seguintes diretórios:
- `EntradasTSP/`: Contém arquivos de dados de teste para o TSP
- `Genetic_Algorithm/`: Implementação do Algoritmo Genético
- `ACO_Algorithm/`: Implementação modularizada da Otimização por Colônia de Formigas

## Algoritmo Genético

A implementação do Algoritmo Genético inclui:

### Estrutura de Classes
- **Individual**: Representa uma solução (rota) para o TSP
  - Métodos para cálculo de fitness, crossover, mutação e 2-opt
  - Suporte à criação de caminhos aleatórios e validação
- **GeneticOptimizer**: Gerencia a evolução da população
  - Suporte à evolução sequencial e paralela
  - Monitoramento de convergência
  - Estratégias adaptativas para seleção e reprodução

### Principais Características
- **Seleção Elitista**: Preserva os melhores indivíduos entre gerações
- **Crossover Ordenado (OX)**: Cria novos indivíduos preservando a ordem das cidades
- **Crossover de Múltiplos Pontos**: Alternativa ao crossover ordenado padrão
- **Mutação por Troca**: Troca aleatória de posições na rota
- **Mutação por Inversão**: Inverte um segmento da rota
- **Heurística 2-opt**: Melhoria local das rotas encontradas
- **Lista Tabu**: Evita ciclos durante a busca
- **Processamento Paralelo**: Acelera o cálculo de fitness e a evolução
  - Implementação multiprocessing com pool de processos
  - Otimização de tamanho de chunks para balanceamento de carga
  - Escalabilidade automática com base no número de CPUs disponíveis

### Visualização e Análise
- **Gráficos de Convergência**: Visualização da evolução do fitness ao longo das gerações
- **Visualização de Rotas**: Plotagem das melhores rotas encontradas
- **Métricas de Desempenho**: Tempo de execução e qualidade da solução

### Otimização Bayesiana de Hiperparâmetros

Recentemente, foi adicionada uma implementação de **Otimização Bayesiana** para encontrar automaticamente os melhores hiperparâmetros para o Algoritmo Genético. Esta abordagem oferece várias vantagens:

1. **Eficiência na busca**: Encontra bons hiperparâmetros com menos avaliações em comparação com métodos de busca em grade ou aleatória
2. **Modelagem probabilística**: Utiliza um processo gaussiano para modelar a função objetivo, permitindo estimar a incerteza
3. **Exploração inteligente**: Equilibra a exploração de áreas desconhecidas e a intensificação em áreas promissoras

#### Hiperparâmetros Otimizados
- **Tamanho da população**: Número de indivíduos em cada geração (50-200)
- **Tamanho da elite**: Proporção dos melhores indivíduos mantidos entre gerações (0.05-0.3)
- **Taxa de mutação**: Probabilidade de aplicar mutação a um indivíduo (0.001-0.1)
- **Taxa de 2-opt**: Probabilidade de aplicar a melhoria 2-opt (0.05-0.3)
- **Taxa de crossover**: Probabilidade de realizar crossover (0.5-0.9)

#### Como Usar a Otimização Bayesiana
```
python Genetic_Algorithm/main.py --modo bayesiano --arquivo Qatar.txt --chamadas 20 --inicios 5 --geracoes_bo 200
```

Onde:
- `--modo bayesiano`: Ativa a otimização bayesiana
- `--arquivo Qatar.txt`: Especifica o arquivo de entrada
- `--chamadas 20`: Define o número total de chamadas da otimização
- `--inicios 5`: Define o número de pontos iniciais aleatórios
- `--geracoes_bo 200`: Define o número de gerações para cada execução de teste

#### Resultados da Otimização
Após a otimização, o algoritmo:
1. Exibe os melhores hiperparâmetros encontrados
2. Plota gráficos de convergência e da influência de cada parâmetro
3. Executa o GA com os parâmetros otimizados por 1000 gerações
4. Exibe a melhor rota e seu custo

### Como Executar o Algoritmo Genético Padrão
```
python Genetic_Algorithm/main.py --modo padrao --arquivo Qatar.txt --geracoes 1000
```

## Otimização por Colônia de Formigas (ACO)

A implementação de ACO foi modularizada e organizada em diferentes componentes para maior manutenibilidade e extensibilidade.

### Estrutura do Módulo ACO

```
ACO_Algorithm/
│
├── __init__.py                 # Torna o pacote importável
├── main.py                     # Arquivo principal de execução com interface CLI
│
├── aco/                        # Módulo para algoritmos ACO
│   ├── __init__.py             # Torna o subpacote importável
│   ├── colony.py               # Implementação principal do algoritmo ACO
│   ├── mmas.py                 # Implementação do Max-Min Ant System
│   └── runner.py               # Funções para execução do ACO (padrão e paralela)
│
├── optimization/               # Módulo para algoritmos de otimização
│   ├── __init__.py             # Torna o subpacote importável
│   ├── k_opt.py                # Implementação do algoritmo K-Opt
│   └── bayesian_optimizer.py   # Otimização bayesiana de hiperparâmetros
│
└── utils/                      # Módulo de utilidades
    ├── __init__.py             # Torna o subpacote importável
    ├── tsp_utils.py            # Funções para manipulação de arquivos TSP e cálculos
    └── visualization.py        # Funções para visualização de soluções
```

### Principais Componentes

#### Algoritmos ACO
- **ACO (Ant Colony Optimization)**: Implementação base do algoritmo
  - Gerenciamento de feromônios e heurísticas
  - Construção probabilística de soluções
  - Atualização de feromônios baseada na qualidade das soluções
- **MMAS (Max-Min Ant System)**: Variante aprimorada que limita os valores de feromônio
  - Prevenção de estagnação com limites dinâmicos
  - Reinicialização de feromônios quando necessário
  - Melhor balanceamento entre exploração e intensificação

#### Otimização
- **K-Opt**: Implementação do algoritmo K-Opt para melhoria local
  - Suporte para diferentes valores de K (2-opt, 3-opt, etc.)
  - Estratégias de busca local para refinamento de soluções
- **Otimização Bayesiana**: Ajuste automático de hiperparâmetros
  - Modelagem probabilística do espaço de parâmetros
  - Exploração inteligente de configurações promissoras

#### Utilitários
- **Funções para TSP**: Leitura de arquivos, cálculo de distâncias
- **Visualização**: Plotagem de rotas, gráficos de convergência

### Como Executar o ACO

A nova estrutura modularizada permite uma execução mais flexível e organizada:

#### Execução Padrão
```
python -m ACO_Algorithm.main --modo padrao --arquivo Qatar.txt --num_formigas 30 --alpha 0.8 --beta 0.8 --rho 0.9 --iteracoes 100 --k_opt 2
```

#### Execução com Otimização Bayesiana
```
python -m ACO_Algorithm.main --modo bayesiano --arquivo Qatar.txt --chamadas 20 --inicios 5 --iteracoes_bo 50 --k_opt 2
```

Parâmetros disponíveis:
- `--modo`: 'padrao' ou 'bayesiano'
- `--arquivo`: Nome do arquivo TSP (ex: Qatar.txt, Djibouti.txt)
- `--num_formigas`: Número de formigas (modo padrão)
- `--alpha`: Importância do feromônio (modo padrão)
- `--beta`: Importância da heurística (modo padrão)
- `--rho`: Taxa de evaporação (modo padrão)
- `--iteracoes`: Número de iterações (modo padrão)
- `--chamadas`: Número de chamadas para otimização bayesiana
- `--inicios`: Número de pontos aleatórios iniciais
- `--iteracoes_bo`: Número de iterações por teste na otimização bayesiana
- `--k_opt`: Valor de k para a otimização K-Opt

## Comparação entre Algoritmos

O projeto permite uma comparação direta entre os diferentes algoritmos implementados (GA e ACO), considerando:

- **Qualidade da solução**: Distância total do melhor caminho encontrado
- **Velocidade de convergência**: Número de iterações necessárias para encontrar boas soluções
- **Tempo de execução**: Eficiência computacional de cada abordagem
- **Robustez**: Consistência dos resultados em diferentes execuções
- **Escalabilidade**: Desempenho em problemas de diferentes tamanhos

## Conjunto de Dados

O repositório inclui diversos conjuntos de dados do TSP, variando de problemas pequenos (como Djibouti com 38 cidades) a problemas maiores. Alguns exemplos:
- Djibouti (38 cidades)
- Qatar (194 cidades)
- Argentina (9 cidades)
- China (71 cidades)
- Japão (124 cidades)
- Tanzânia (127 cidades)
- E outros mais...

## Requisitos

- Python 3.6+
- NumPy
- Matplotlib
- scikit-optimize (para otimização bayesiana)
- tqdm (para barras de progresso)

## Instalação

Para instalar todas as dependências necessárias:

```
pip install numpy matplotlib scikit-optimize pandas tqdm
```

## Resultados

Os algoritmos foram testados em vários conjuntos de dados e comparados em termos de:
- Qualidade da solução (distância total)
- Tempo de execução
- Convergência

A otimização bayesiana melhorou significativamente o desempenho tanto do Algoritmo Genético quanto do ACO, encontrando automaticamente configurações de hiperparâmetros que resultam em melhores soluções com menos gerações ou iterações.
