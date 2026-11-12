"""Primeira olhada nos clientes e nas compras com cartão da Loja Exemplo.

Rode de dentro desta pasta:

    python3 primeira_olhada.py

São três contas: quanto cada coluna pesa na distância entre clientes, cinco grupos
de clientes feitos com e sem escala, e uma fila das 200 compras mais estranhas.
A coluna de fraude não entra em nenhuma conta; ela só aparece no fim, para ver
quantas da fila eram fraude.
"""
from itertools import permutations

import numpy as np
import pandas as pd

COLUNAS = ["pedidos_12m", "gasto_12m", "ticket_medio", "dias_desde_ultima_compra",
           "categorias_distintas", "pct_pedidos_cupom", "pct_pedidos_app", "devolucoes_12m"]


def pct(x):
    return f"{x:.1f}".replace(".", ",") + "%"


def reais(x):
    return f"{x:,.2f}".replace(",", "_").replace(".", ",").replace("_", ".")


def kmeans(X, k, partida):
    """K-means simples: escolhe k centros espalhados e repete 'cada um no centro mais perto'."""
    rng = np.random.default_rng(partida)
    centros = [X[rng.integers(len(X))]]
    for _ in range(k - 1):
        d = ((X[:, None, :] - np.array(centros)[None]) ** 2).sum(axis=2).min(axis=1)
        centros.append(X[rng.choice(len(X), p=d / d.sum())])
    centros = np.array(centros)
    for _ in range(300):
        grupo = ((X[:, None, :] - centros[None]) ** 2).sum(axis=2).argmin(axis=1)
        novos = np.array([X[grupo == j].mean(axis=0) for j in range(k)])
        if np.allclose(novos, centros):
            break
        centros = novos
    return grupo


def trocam(a, b, k):
    """Quantos clientes mudam de grupo, depois de casar os números dos grupos da melhor forma."""
    juntos = np.zeros((k, k), dtype=int)
    np.add.at(juntos, (a, b), 1)
    mesmos = max(sum(juntos[i, p[i]] for i in range(k)) for p in permutations(range(k)))
    return len(a) - mesmos


clientes = pd.read_csv("../dados/clientes/clientes.csv")
compras = pd.read_csv("../dados/fraude/transacoes.csv")

print("1. Quanto cada coluna pesa na distância entre clientes, sem escala")
variancia = clientes[COLUNAS].var(ddof=0)
print(f"   {len(clientes)} clientes, {len(COLUNAS)} colunas de compra")
print(f"   gasto_12m: {pct(100 * variancia['gasto_12m'] / variancia.sum())} da variância somada")
print(f"   as outras 7 colunas juntas: {pct(100 * variancia.drop('gasto_12m').sum() / variancia.sum())}")
com_id = clientes[["id_cliente"] + COLUNAS].var(ddof=0)
print(f"   pondo id_cliente junto, ele fica com {pct(100 * com_id['id_cliente'] / com_id.sum())};"
      f" as 7 colunas, com {pct(100 * com_id.drop(['id_cliente', 'gasto_12m']).sum() / com_id.sum())}")

print("2. Cinco grupos de clientes com K-means")
X = clientes[COLUNAS].to_numpy(float)
sem_escala = kmeans(X, 5, partida=0)
print("   sem escala, do grupo com menor gasto para o maior:")
ordem = clientes.groupby(sem_escala)["gasto_12m"].median().sort_values().index
for j in ordem:
    g = clientes[sem_escala == j]
    print(f"     {len(g):>5} clientes   gasto de R$ {reais(g['gasto_12m'].min()):>9}"
          f" a R$ {reais(g['gasto_12m'].max()):>9}")
Z = (X - X.mean(axis=0)) / X.std(axis=0)
partida_a = kmeans(Z, 5, partida=0)
partida_b = kmeans(Z, 5, partida=1)
tamanhos = lambda g: " ".join(str(n) for n in sorted(np.bincount(g, minlength=5), reverse=True))
print(f"   com escala z, um ponto de partida: {tamanhos(partida_a)}")
print(f"   com escala z, outro ponto de partida: {tamanhos(partida_b)}")
n = trocam(partida_a, partida_b, 5)
print(f"   clientes que trocam de grupo entre os dois: {n} ({pct(100 * n / len(clientes))})")

print("3. Fila das 200 compras com cartão mais estranhas")
colunas_compra = ["valor", "itens", "idade_conta_dias", "tentativas_cartao", "distancia_entrega_km"]
C = compras[colunas_compra].to_numpy(float)
estranheza = np.abs((C - C.mean(axis=0)) / C.std(axis=0)).sum(axis=1)
fila = compras.iloc[np.argsort(-estranheza, kind="stable")[:200]]
fraudes = int(compras["fraude"].sum())
na_fila = int(fila["fraude"].sum())
print(f"   {len(compras)} compras; estranheza = soma das distâncias à média, em desvios padrão,"
      f" de {len(colunas_compra)} colunas")
print(f"   na fila: {na_fila} fraudes e {200 - na_fila} compras legítimas")
print(f"   fora da fila: {fraudes - na_fila} das {fraudes} fraudes")
