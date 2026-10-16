import argparse
import copy
import csv
from pathlib import Path

import ale_py
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np


gym.register_envs(ale_py)

ENV_ID = "ALE/Freeway-v5"
TAMANHO_POPULACAO = 20
NUM_GERACOES = 30
NUM_ENTRADAS = 128
NUM_OCULTOS = 16
NUM_ACOES = 3
TAXA_MUTACAO = 0.10
INTENSIDADE_MUTACAO = 0.20
NUM_ELITE = 2
PASTA_RESULTADOS = Path(__file__).resolve().parent / "resultados"


def criar_ambiente(render_mode=None):
	return gym.make(ENV_ID, obs_type="ram", render_mode=render_mode)


class Agente:
	def __init__(self):
		self.W1 = np.random.randn(NUM_ENTRADAS, NUM_OCULTOS)
		self.b1 = np.random.randn(NUM_OCULTOS)
		self.W2 = np.random.randn(NUM_OCULTOS, NUM_ACOES)
		self.b2 = np.random.randn(NUM_ACOES)
		self.fitness = 0.0

	def agir(self, estado):
		estado = estado.astype(np.float32) / 255.0
		saida_oculta = np.tanh(estado @ self.W1 + self.b1)
		saida = saida_oculta @ self.W2 + self.b2
		return int(np.argmax(saida))


def avaliar(agente, render_mode=None):
	env = criar_ambiente(render_mode)
	try:
		estado, informacoes = env.reset()
		fitness = 0.0
		terminado = False
		truncado = False

		while not terminado and not truncado:
			action = agente.agir(estado)
			estado, reward, terminado, truncado, informacoes = env.step(action)
			fitness += reward

		return float(fitness)
	finally:
		env.close()


def avaliar_aleatorio():
	env = criar_ambiente()
	try:
		estado, informacoes = env.reset()
		fitness = 0.0
		terminado = False
		truncado = False

		while not terminado and not truncado:
			action = env.action_space.sample()
			estado, reward, terminado, truncado, informacoes = env.step(action)
			fitness += reward

		return float(fitness)
	finally:
		env.close()


def criar_populacao():
	return [Agente() for indice in range(TAMANHO_POPULACAO)]


def selecionar(populacao):
	populacao_ordenada = sorted(populacao, key=lambda agente: agente.fitness, reverse=True)
	quantidade_pais = max(1, len(populacao_ordenada) // 2)
	return populacao_ordenada[:quantidade_pais]


def mutar(agente, taxa_mutacao=TAXA_MUTACAO, intensidade_mutacao=INTENSIDADE_MUTACAO):
	for parametro in (agente.W1, agente.b1, agente.W2, agente.b2):
		mascara = np.random.rand(*parametro.shape) < taxa_mutacao
		parametro += mascara * np.random.randn(*parametro.shape) * intensidade_mutacao


def nova_geracao(populacao, taxa_mutacao=TAXA_MUTACAO, intensidade_mutacao=INTENSIDADE_MUTACAO):
	selecionados = selecionar(populacao)
	populacao_ordenada = sorted(populacao, key=lambda agente: agente.fitness, reverse=True)
	nova_populacao = [copy.deepcopy(agente) for agente in populacao_ordenada[:NUM_ELITE]]

	while len(nova_populacao) < TAMANHO_POPULACAO:
		pai = selecionados[np.random.randint(0, len(selecionados))]
		filho = copy.deepcopy(pai)
		mutar(filho, taxa_mutacao, intensidade_mutacao)
		nova_populacao.append(filho)

	return nova_populacao


def evoluir(taxa_mutacao=TAXA_MUTACAO, intensidade_mutacao=INTENSIDADE_MUTACAO):
	populacao = criar_populacao()
	historico = []

	for geracao in range(NUM_GERACOES):
		for agente in populacao:
			agente.fitness = avaliar(agente)

		valores = [agente.fitness for agente in populacao]
		registro = {
			"geracao": geracao,
			"melhor": max(valores),
			"media": float(np.mean(valores)),
			"pior": min(valores),
		}
		historico.append(registro)
		print(
			f"Geração {geracao:03d} | "
			f"Melhor: {registro['melhor']:.2f} | "
			f"Média: {registro['media']:.2f} | "
			f"Pior: {registro['pior']:.2f}"
		)
		melhor_agente = copy.deepcopy(max(populacao, key=lambda agente: agente.fitness))
		populacao = nova_geracao(populacao, taxa_mutacao, intensidade_mutacao)

	return historico, melhor_agente


def salvar_historico(historico, nome_arquivo="fitness.csv"):
	PASTA_RESULTADOS.mkdir(exist_ok=True)
	with (PASTA_RESULTADOS / nome_arquivo).open("w", newline="", encoding="utf-8") as arquivo:
		escritor = csv.DictWriter(arquivo, fieldnames=["geracao", "melhor", "media", "pior"])
		escritor.writeheader()
		escritor.writerows(historico)


def gerar_grafico(historico, nome_arquivo="fitness.png"):
	geracoes = [registro["geracao"] for registro in historico]
	melhores = [registro["melhor"] for registro in historico]
	medias = [registro["media"] for registro in historico]

	plt.figure()
	plt.plot(geracoes, melhores, label="Melhor fitness")
	plt.plot(geracoes, medias, label="Fitness médio")
	plt.xlabel("Geração")
	plt.ylabel("Fitness")
	plt.title("Evolução do agente no Freeway")
	plt.legend()
	plt.grid(True)
	plt.tight_layout()
	PASTA_RESULTADOS.mkdir(exist_ok=True)
	plt.savefig(PASTA_RESULTADOS / nome_arquivo)
	plt.close()


def executar_experimentos():
	experimentos = {
		"experimento_a": (0.01, 0.05),
		"experimento_b": (0.10, 0.20),
		"experimento_c": (0.40, 0.50),
	}
	for nome, (taxa, intensidade) in experimentos.items():
		print(f"\n{nome}: taxa={taxa:.2f}, intensidade={intensidade:.2f}")
		historico, melhor_agente = evoluir(taxa, intensidade)
		salvar_historico(historico, f"{nome}.csv")
		gerar_grafico(historico, f"{nome}.png")


def main():
	parser = argparse.ArgumentParser(description="Neuroevolução aplicada ao Atari Freeway.")
	parser.add_argument("--experimentos", action="store_true", help="Executa A, B e C.")
	parser.add_argument("--renderizar", action="store_true", help="Renderiza o melhor agente.")
	args = parser.parse_args()

	baseline = avaliar_aleatorio()
	print(f"Baseline aleatório | Fitness: {baseline:.2f}")

	if args.experimentos:
		executar_experimentos()
		return

	historico, melhor_agente = evoluir()
	salvar_historico(historico)
	gerar_grafico(historico)
	print(f"Melhor agente evoluído | Fitness: {melhor_agente.fitness:.2f}")
	if args.renderizar:
		print(f"Melhor agente renderizado | Fitness: {avaliar(melhor_agente, 'human'):.2f}")


if __name__ == "__main__":
	main()
