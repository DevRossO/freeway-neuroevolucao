# Neuroevolução Aplicada ao Atari Freeway

Projeto desenvolvido para a disciplina de **Inteligência Artificial**.

## Visão geral

Este projeto implementa e demonstra um processo evolutivo, combinando um algoritmo genético com uma pequena rede neural, para treinar um agente capaz de tomar decisões no jogo *Freeway*, do Atari 2600.

O agente utiliza diretamente os dados da memória RAM do Atari. O treinamento não utiliza aprendizagem por reforço tradicional, como DQN ou Q-Learning, nem *backpropagation*.

## Ambiente e requisitos

### Dependências

Instale as bibliotecas necessárias com:

```bash
pip install "gymnasium[atari]" ale-py numpy matplotlib
```

### Estrutura do projeto

```text
freeway-neuroevolucao/
├── main.py              # Lógica do agente e do algoritmo evolutivo
├── README.md            # Documentação do projeto
└── resultados/          # Arquivos CSV e gráficos gerados
    ├── experimento_a.csv
    ├── experimento_a.png
    ├── experimento_b.csv
    ├── experimento_b.png
    ├── experimento_c.csv
    └── experimento_c.png
```

## Parâmetros do algoritmo

- **Ambiente:** `ALE/Freeway-v5`.
- **Observação:** memória RAM do Atari, com 128 valores.
- **Tamanho da população:** 20 indivíduos.
- **Número de gerações:** 30.
- **Arquitetura da rede:** 128 entradas, 16 neurônios ocultos com `tanh` e 3 saídas.
- **Ações:** `0 = NOOP`, `1 = UP` e `2 = DOWN`.
- **Seleção:** os 50% melhores indivíduos da geração atual.
- **Elitismo:** preservação direta dos 2 melhores indivíduos, sem mutação.

Os valores podem ser alterados diretamente no início de `main.py`.

## Como executar

O script utiliza `argparse` para permitir diferentes formas de execução.

### Execução padrão

Executa a evolução com os parâmetros padrão e salva o histórico em CSV e o gráfico em `resultados/`.

```bash
python main.py
```

### Execução com renderização

Executa a evolução e, ao final, abre a janela do emulador para exibir o melhor agente.

```bash
python main.py --renderizar
```

### Execução dos três experimentos

Executa automaticamente os experimentos A, B e C, salvando os respectivos arquivos CSV e PNG.

```bash
python main.py --experimentos
```

## Resultados dos experimentos

### Baseline aleatório

- **Fitness alcançado:** `0.00`.
- **Comportamento:** o agente escolhe aleatoriamente entre subir, descer e ficar parado. Nesse cenário, a galinha não concluiu nenhuma travessia.

### Tabela comparativa

| Experimento | Taxa de mutação | Intensidade | Fitness inicial (G0) | Fitness final (G29) | Maior fitness | Média final (G29) | Pior indivíduo (G29) |
|:--|--:|--:|--:|--:|--:|--:|--:|
| A (baixa) | 1% (`0.01`) | `0.05` | `23.00` | `27.00` | `27.00` | `22.65` | `20.00` |
| B (média) | 10% (`0.10`) | `0.20` | `25.00` | `28.00` | `28.00` | `24.05` | `21.00` |
| C (alta) | 40% (`0.40`) | `0.50` | `21.00` | `26.00` | `27.00` | `20.95` | `11.00` |

## Análise dos gráficos

- **Experimento A:** a baixa mutação causou alterações pequenas nos descendentes. O progresso foi lento e a população se manteve próxima dos patamares iniciais.
- **Experimento B:** apresentou o melhor equilíbrio entre exploração e preservação das soluções. Alcançou o maior fitness, `28.00`, e a maior média populacional final, `24.05`.
- **Experimento C:** a mutação agressiva provocou alta variabilidade. A alteração intensa dos pesos prejudicou soluções funcionais herdadas dos pais e reduziu a média da população.

## Análise científica e respostas às perguntas

### 1. O que é o agente e qual estado ele recebe do ambiente?

O agente é uma instância da classe `Agente`, definida em `main.py`. Ele possui os parâmetros `W1`, `b1`, `W2` e `b2`, que formam a rede neural.

O ambiente envia um vetor com 128 valores da memória RAM do Atari. No método `agir(estado)`, esses valores, originalmente entre 0 e 255, são normalizados dividindo-os por `255.0` antes de serem processados pela rede.

### 2. Qual é a diferença entre ação, recompensa e fitness?

- **Ação:** decisão tomada pelo agente em cada passo: `0 = NOOP`, `1 = UP` ou `2 = DOWN`.
- **Recompensa:** sinal retornado pelo ambiente depois de uma ação. No Freeway, o ambiente fornece `1.0` quando a galinha alcança o topo e `0.0` nos demais passos.
- **Fitness:** valor acumulado das recompensas durante todo o episódio. É a métrica usada pelo algoritmo genético para comparar os agentes.

### 3. Como indivíduo, população e geração aparecem no código?

- **Indivíduo:** cada objeto criado a partir de `Agente()`.
- **População:** lista de agentes criada pela função `criar_populacao()`.
- **Geração:** cada iteração do laço `for geracao in range(NUM_GERACOES)` dentro da função `evoluir()`.

### 4. Como a seleção foi implementada e por que esse critério foi escolhido?

A função `selecionar()` ordena os agentes do maior para o menor fitness e seleciona a metade superior da população.

Esse critério cria pressão seletiva, pois os indivíduos com melhor desempenho têm maior chance de gerar descendentes, mas ainda preserva parte da diversidade da população.

### 5. O que aconteceria se não existisse elitismo?

A função `nova_geracao()` copia diretamente os 2 melhores agentes usando `copy.deepcopy`, sem aplicar mutação. Sem elitismo, uma mutação poderia alterar negativamente o melhor indivíduo e fazer o algoritmo perder uma solução importante.

### 6. Como a taxa e a intensidade de mutação afetam os descendentes?

- **Taxa de mutação:** define a probabilidade de cada peso sofrer alteração.
- **Intensidade de mutação:** define a escala do ruído aleatório adicionado aos pesos selecionados.

Uma taxa muito baixa limita a exploração. Uma taxa muito alta pode destruir informações úteis herdadas dos pais. No experimento realizado, o valor intermediário do experimento B produziu o melhor resultado geral.

### 7. Onde ocorre a exploração no algoritmo?

A exploração ocorre na criação da população inicial, quando os pesos são sorteados aleatoriamente com `np.random.randn`, e durante a mutação dos descendentes, quando pequenas perturbações são aplicadas aos pesos e aos *biases*.

### 8. Os resultados indicam convergência? Qual evidência sustenta a resposta?

Sim. No experimento B, a média da população aumentou de `7.70` na geração 0 para aproximadamente `24.00` e se manteve estável até a geração 29. O melhor fitness também se estabilizou em `28.00`.

Essa estabilização das curvas de melhor fitness e fitness médio indica convergência da população para soluções semelhantes.

### 9. O melhor agente superou o agente aleatório? Qual evidência mostra isso?

Sim.

- **Agente aleatório:** fitness `0.00`, sem travessias concluídas.
- **Melhor agente evoluído no experimento B:** fitness `28.00`.

A diferença entre esses valores demonstra que o processo evolutivo produziu um agente significativamente melhor que o comportamento aleatório.

### 10. Qual mudança poderia melhorar o resultado?

Algumas possibilidades são:

1. **Implementar crossover:** combinar partes dos pesos de dois progenitores, em vez de utilizar somente cópia e mutação.
2. **Filtrar as entradas da RAM:** utilizar apenas os endereços relacionados à posição da galinha e dos carros, reduzindo o ruído dos 128 valores.
3. **Utilizar mutação adaptativa:** reduzir gradualmente a taxa e a intensidade de mutação ao longo das gerações, permitindo ajustes mais finos no final do treinamento.
