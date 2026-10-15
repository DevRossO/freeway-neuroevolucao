import copy
import csv
import os

import ale_py
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


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
	# Cria o ambiente e recebe o estado inicial do episódio.
	env = criar_ambiente()
	espaco_acoes = env.action_space
	estado, informacoes = env.reset()
	fitness = 0.0
	terminado = False
	truncado = False

	# O episódio termina quando terminated ou truncated for True.
	while not terminado and not truncado:
		action = agente.agir(estado)
		estado, reward, terminado, truncado, informacoes = env.step(action)
		fitness = fitness + reward

	env.close()
	return fitness


def criar_populacao():
	populacao = []

	for indice in range(TAMANHO_POPULACAO):
		populacao.append(Agente())

	return populacao


def obter_fitness(agente):
	return agente.fitness


def selecionar(populacao):
	# Ordena do maior para o menor fitness.
	populacao_ordenada = sorted(
		populacao,
		key=obter_fitness,
		reverse=True,
	)

	quantidade_pais = len(populacao_ordenada) // 2
	if quantidade_pais < 1:
		quantidade_pais = 1

	return populacao_ordenada[:quantidade_pais]


def mutar(agente):
	# Cada posição sorteada recebe uma pequena alteração aleatória.
	mascara_W1 = np.random.rand(*agente.W1.shape) < TAXA_MUTACAO
	mascara_b1 = np.random.rand(*agente.b1.shape) < TAXA_MUTACAO
	mascara_W2 = np.random.rand(*agente.W2.shape) < TAXA_MUTACAO
	mascara_b2 = np.random.rand(*agente.b2.shape) < TAXA_MUTACAO

	agente.W1 += mascara_W1 * np.random.randn(*agente.W1.shape) * INTENSIDADE_MUTACAO
	agente.b1 += mascara_b1 * np.random.randn(*agente.b1.shape) * INTENSIDADE_MUTACAO
	agente.W2 += mascara_W2 * np.random.randn(*agente.W2.shape) * INTENSIDADE_MUTACAO
	agente.b2 += mascara_b2 * np.random.randn(*agente.b2.shape) * INTENSIDADE_MUTACAO


def nova_geracao(populacao):
	# Seleciona os melhores indivíduos para serem pais.
	selecionados = selecionar(populacao)
	populacao_ordenada = sorted(
		populacao,
		key=obter_fitness,
		reverse=True,
	)

	nova_populacao = []

	# O elitismo preserva os melhores sem mutação.
	for indice in range(NUM_ELITE):
		elite = copy.deepcopy(populacao_ordenada[indice])
		nova_populacao.append(elite)

	# Completa a população com cópias mutadas dos pais selecionados.
	while len(nova_populacao) < TAMANHO_POPULACAO:
		indice_pai = np.random.randint(0, len(selecionados))
		filho = copy.deepcopy(selecionados[indice_pai])
		mutar(filho)
		nova_populacao.append(filho)

	return nova_populacao


def salvar_historico(historico):
	os.makedirs("resultados", exist_ok=True)

	with open("resultados/fitness.csv", "w", newline="", encoding="utf-8") as arquivo:
		escritor = csv.DictWriter(
			arquivo,
			fieldnames=["geracao", "melhor", "media", "pior"],
		)
		escritor.writeheader()
		escritor.writerows(historico)


def gerar_grafico(historico):
	geracoes = []
	melhores = []
	medias = []

	for registro in historico:
		geracoes.append(registro["geracao"])
		melhores.append(registro["melhor"])
		medias.append(registro["media"])

	plt.plot(geracoes, melhores, label="Melhor fitness")
	plt.plot(geracoes, medias, label="Fitness médio")
	plt.xlabel("Geração")
	plt.ylabel("Fitness")
	plt.title("Evolução do agente no Freeway")
	plt.legend()
	plt.grid(True)
	plt.savefig("resultados/fitness.png")
	plt.close()


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


salvar_historico(historico)
gerar_grafico(historico)