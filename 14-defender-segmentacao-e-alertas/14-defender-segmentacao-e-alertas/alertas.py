"""Fila de revisão das compras com cartão e as contas que vão para o relatório.

A nota é a da aula «Priorizar suspeitas de fraude» (nota.py, copiado sem mudança): aprendida
com janeiro a março e aplicada dia a dia de abril a junho. A coluna `fraude` não entra na
nota nem na fila; ela só entra para contar, depois, como se a contestação tivesse chegado.
"""
import numpy as np
import pandas as pd

from nota import ajustar, pontuar

DADOS = '../dados/fraude/transacoes.csv'
INICIO = '2026-04-01'
SORTEIO = 20260401


def carregar(caminho=DADOS):
    compras = pd.read_csv(caminho)
    compras['dia'] = compras['data_hora'].str[:10]
    return compras


def com_nota(compras, inicio_ref='2026-01-01', faixas=10, escala=None):
    """Compras de abril em diante com a coluna `nota`, aprendida de `inicio_ref` até março."""
    referencia = compras[(compras['dia'] >= inicio_ref) & (compras['dia'] < INICIO)]
    novas = compras[compras['dia'] >= INICIO].copy()
    novas['nota'] = pontuar(ajustar(referencia, faixas=faixas, escala=escala), novas)['nota']
    return novas


def montar_fila(compras, topo, sorteadas, partida=SORTEIO, por='nota'):
    """Para cada dia, as `topo` de maior `por` e mais `sorteadas` entre as outras.

    Empate: entra primeiro a compra mais antiga. Anota a origem e a chance de cada revisada.
    """
    if topo < 0 or sorteadas < 0:
        raise ValueError('topo e sorteadas não podem ser negativos')
    if 'fraude' in compras.columns:
        raise ValueError('tire a coluna fraude antes de montar a fila')
    if compras[por].isna().any():
        raise ValueError(f'há compras sem {por!r}')
    rng = np.random.default_rng(partida)
    partes = []
    for _, dia in compras.groupby('dia', sort=True):
        ordem = dia.sort_values([por, 'id_transacao'], ascending=[False, True])
        cima = ordem.head(topo).assign(origem='topo', chance=1.0)
        resto = ordem.iloc[len(cima):]
        k = min(sorteadas, len(resto))
        escolha = sorted(rng.choice(len(resto), size=k, replace=False)) if k else []
        baixo = resto.iloc[escolha].assign(origem='sorteada', chance=k / len(resto) if len(resto) else 0.0)
        partes += [cima, baixo]
    return pd.concat(partes)


def wilson(k, n, z=1.959964):
    """Margem de 95% de Wilson para uma proporção k/n. Retorna (de, até)."""
    if n <= 0 or not 0 <= k <= n:
        raise ValueError('precisa de 0 <= k <= n e n > 0')
    p = k / n
    den = 1 + z * z / n
    centro = (p + z * z / (2 * n)) / den
    meia = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, centro - meia), min(1.0, centro + meia)
