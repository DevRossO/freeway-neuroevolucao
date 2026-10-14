import copy

import ale_py
import gymnasium as gym
import numpy as np


gym.register_envs(ale_py)


# ---------------------------------------------------------
# CONFIGURAÇÕES
# ---------------------------------------------------------
ENV_ID = "ALE/Freeway-v5"
TAMANHO_POPULACAO = 20
NUM_GERACOES = 30
NUM_ENTRADAS = 128
NUM_OCULTOS = 16
NUM_ACOES = 3
TAXA_MUTACAO = 0.10
INTENSIDADE_MUTACAO = 0.20
NUM_ELITE = 2


def criar_ambiente(render_mode=None):
	return gym.make(
		ENV_ID,
		obs_type="ram",
		render_mode=render_mode,
	)


class Agente:
	def __init__(self):
		# Cada indivíduo começa com parâmetros diferentes.
		self.W1 = np.random.randn(NUM_ENTRADAS, NUM_OCULTOS)
		self.b1 = np.random.randn(NUM_OCULTOS)
		self.W2 = np.random.randn(NUM_OCULTOS, NUM_ACOES)
		self.b2 = np.random.randn(NUM_ACOES)
		self.fitness = 0.0

	def agir(self, estado):
		# RAM: valores inteiros entre 0 e 255.
		estado = estado.astype(np.float32) / 255.0
		oculto = np.tanh(estado @ self.W1 + self.b1)
		saida = oculto @ self.W2 + self.b2
		# A maior saída define a ação escolhida.
		return int(np.argmax(saida))


def avaliar(agente):
	# TODO - AGENTE + AMBIENTE + AÇÃO + RECOMPENSA + FITNESS
	# 1. Criar/reiniciar o ambiente.
	# 2. Receber o estado inicial.
	# 3. Enquanto o episódio não terminar:
	#    a) pedir uma ação ao agente;
	#    b) executar a ação no ambiente;
	#    c) receber novo estado e reward;
	#    d) acumular a recompensa.
	# 4. Fechar o ambiente.
	# 5. Retornar o fitness do indivíduo.
	pass


def criar_populacao():
	# TODO - INDIVÍDUO + POPULAÇÃO
	# Criar TAMANHO_POPULACAO agentes independentes.
	# Todos devem começar com parâmetros aleatórios próprios.
	pass


def selecionar(populacao):
	# TODO - SELEÇÃO
	# Escolher os indivíduos que poderão gerar descendentes.
	# O grupo deverá definir e explicar seu critério.
	pass


def mutar(agente):
	# TODO - MUTAÇÃO + EXPLORAÇÃO
	# Alterar aleatoriamente parte de W1, b1, W2 e b2.
	# Utilizar obrigatoriamente:
	# TAXA_MUTACAO
	# INTENSIDADE_MUTACAO
	pass


def nova_geracao(populacao):
	# TODO - ELITISMO + REPRODUÇÃO + MUTAÇÃO
	# 1. Selecionar os pais.
	# 2. Copiar NUM_ELITE melhores sem alteração.
	# 3. Criar descendentes a partir dos selecionados.
	# 4. Aplicar mutação nos descendentes.
	# 5. Retornar exatamente TAMANHO_POPULACAO agentes.
	pass


populacao = criar_populacao()
historico = []


for geracao in range(NUM_GERACOES):
	# Avaliar TODOS os indivíduos da geração.
	for agente in populacao:
		agente.fitness = avaliar(agente)

	valores = [agente.fitness for agente in populacao]
	melhor = max(valores)
	media = float(np.mean(valores))
	pior = min(valores)
	historico.append(
		{
			"geracao": geracao,
			"melhor": melhor,
			"media": media,
			"pior": pior,
		}
	)
	print(
		f"Geração {geracao:03d} | "
		f"Melhor: {melhor:.2f} | "
		f"Média: {media:.2f} | "
		f"Pior: {pior:.2f}"
	)

	# Produzir a próxima geração.
	populacao = nova_geracao(populacao)


# TODO:
# salvar o histórico em CSV
# gerar o gráfico de evolução
# executar/renderizar o melhor agente encontrado