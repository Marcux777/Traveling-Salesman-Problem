# Problema do Caixeiro Viajante (Traveling Salesman Problem)

Este repositório contém implementações de diferentes algoritmos para resolver o Problema do Caixeiro Viajante (TSP), incluindo:
- Algoritmo Genético (GA)
- Otimização por Colônia de Formigas (ACO)
- K-opt local search

## Estrutura do Projeto

O projeto está organizado nos seguintes diretórios:
- `EntradasTSP/`: Contém arquivos de dados de teste para o TSP
- `Genetic_Algorithm/`: Implementação do Algoritmo Genético
- `ACO + K-opt/`: Implementação da Otimização por Colônia de Formigas com K-opt

## Algoritmo Genético

A implementação do Algoritmo Genético inclui:

### Estrutura de Classes
- **Individual**: Representa uma solução (rota) para o TSP
- **GeneticOptimizer**: Gerencia a evolução da população

### Principais Características
- **Seleção Elitista**: Preserva os melhores indivíduos entre gerações
- **Crossover Ordenado (OX)**: Cria novos indivíduos preservando a ordem das cidades
- **Crossover de Múltiplos Pontos**: Alternativa ao crossover ordenado padrão
- **Mutação por Troca**: Troca aleatória de posições na rota
- **Mutação por Inversão**: Inverte um segmento da rota
- **Heurística 2-opt**: Melhoria local das rotas encontradas
- **Lista Tabu**: Evita ciclos durante a busca
- **Processamento Paralelo**: Acelera o cálculo de fitness e a evolução

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

A implementação de ACO inclui:

### Principais Características
- **Regra de atualização de feromônios**: Deposição e evaporação de feromônios
- **Heurística da visibilidade**: Utiliza o inverso da distância para guiar as formigas
- **MMAS (Max-Min Ant System)**: Variante que limita a quantidade de feromônio para evitar convergência prematura
- **Processamento Paralelo**: Executa múltiplas instâncias em paralelo para encontrar melhores soluções

### Melhoria K-opt
A implementação inclui uma melhoria K-opt para otimizar localmente as melhores rotas encontradas, incluindo:
- **2-opt**: Troca cruzada de duas arestas
- **3-opt**: Troca cruzada de três arestas
- **K-opt (K>3)**: Implementação simplificada para valores maiores de K

### Otimização Bayesiana de Hiperparâmetros para ACO

Assim como no GA, foi implementada a **Otimização Bayesiana** para o ACO, permitindo encontrar automaticamente os melhores valores para os principais hiperparâmetros:

#### Hiperparâmetros Otimizados para ACO
- **Número de formigas**: Quantidade de formigas na colônia (5-50)
- **Alpha (α)**: Importância dos feromônios na decisão de rota (0.1-2.0)
- **Beta (β)**: Importância da heurística (distância) na decisão de rota (0.1-5.0)
- **Rho (ρ)**: Taxa de evaporação dos feromônios (0.1-0.99)

#### Como Usar a Otimização Bayesiana para ACO
```
python "ACO + K-opt/ACO_main.py" --modo bayesiano --arquivo Djibouti.txt --chamadas 20 --inicios 5 --iteracoes_bo 50 --k_opt 2
```

Onde:
- `--modo bayesiano`: Ativa a otimização bayesiana
- `--arquivo Djibouti.txt`: Especifica o arquivo de entrada
- `--chamadas 20`: Define o número total de chamadas da otimização
- `--inicios 5`: Define o número de pontos iniciais aleatórios
- `--iteracoes_bo 50`: Define o número de iterações de ACO para cada teste
- `--k_opt 2`: Define o valor de k para o algoritmo K-opt

#### Resultados da Otimização
O processo de otimização:
1. Testa várias configurações de parâmetros, aprendendo a cada iteração
2. Exibe os melhores hiperparâmetros encontrados
3. Plota gráficos de convergência e da influência de cada parâmetro
4. Executa o ACO com os parâmetros otimizados por 200 iterações
5. Aplica K-opt na melhor solução encontrada
6. Exibe a melhor rota e seu custo, junto com uma visualização gráfica

### Como Executar o ACO Padrão
```
python "ACO + K-opt/ACO_main.py" --modo padrao --arquivo Djibouti.txt --num_formigas 30 --alpha 0.8 --beta 0.8 --rho 0.9 --iteracoes 100 --k_opt 2
```

## Conjunto de Dados

O repositório inclui diversos conjuntos de dados do TSP, variando de problemas pequenos (como Djibouti com 38 cidades) a problemas maiores. Alguns exemplos:
- Djibouti (38 cidades)
- Qatar (194 cidades)
- Argentina (9 cidades)
- China (71 cidades)
- E outros mais...

## Requisitos

- Python 3.6+
- NumPy
- Matplotlib
- scikit-optimize (para otimização bayesiana)

## Resultados

Os algoritmos foram testados em vários conjuntos de dados e comparados em termos de:
- Qualidade da solução (distância total)
- Tempo de execução
- Convergência

A otimização bayesiana melhorou significativamente o desempenho tanto do Algoritmo Genético quanto do ACO, encontrando automaticamente configurações de hiperparâmetros que resultam em melhores soluções com menos gerações ou iterações.
