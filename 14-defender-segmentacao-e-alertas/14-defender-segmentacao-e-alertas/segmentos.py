"""Grupos de clientes com K-means e as contas que vão para o relatório.

Só usa o pandas e o NumPy. O K-means é o mesmo da aula 00, agora com 10 pontos de partida
por rodada: fica o arranjo com a menor soma das distâncias ao centro.
"""
from itertools import permutations

import numpy as np
import pandas as pd

FORA = {
    'id_cliente': 'é a ordem do cadastro, não um jeito de comprar',
    'data_cadastro': 'é uma data; transforme em número de propósito antes de usar',
}


def preparar(clientes, colunas, log=()):
    """Matriz pronta para medir distância: log nas colunas pedidas e escala z em todas.

    Recusa o id, coluna de texto, coluna com vazio e coluna que não varia. O vazio precisa
    de uma decisão sua antes (tirar a coluna, preencher, ou marcar quem não tem valor).
    """
    if not colunas:
        raise ValueError('nenhuma coluna escolhida')
    for c in colunas:
        if c in FORA:
            raise ValueError(f'{c} {FORA[c]}')
        if c not in clientes.columns:
            raise ValueError(f'a coluna {c!r} não existe')
        if not pd.api.types.is_numeric_dtype(clientes[c]):
            raise ValueError(f'{c} não é número')
        if clientes[c].isna().any():
            raise ValueError(f'{c} tem {int(clientes[c].isna().sum())} vazios; decida o que fazer com eles antes')
    faltou = [c for c in log if c not in colunas]
    if faltou:
        raise ValueError(f'log pedido em coluna que não entrou: {faltou}')
    X = clientes[colunas].astype(float).copy()
    for c in log:
        if (X[c] < 0).any():
            raise ValueError(f'{c} tem valor negativo; log1p não serve')
        X[c] = np.log1p(X[c])
    X = X.to_numpy()
    desvio = X.std(axis=0)
    if (desvio == 0).any():
        raise ValueError('uma coluna não varia: ' + ', '.join(c for c, d in zip(colunas, desvio) if d == 0))
    return (X - X.mean(axis=0)) / desvio


def _distancias(Z, centros):
    return ((Z[:, None, :] - centros[None]) ** 2).sum(axis=2)


def kmeans(Z, k, partida, n_init=10):
    """Retorna (grupo de cada linha, centros). Mesma partida, mesmo resultado."""
    if k < 1 or k > len(Z):
        raise ValueError('k precisa estar entre 1 e o número de linhas')
    rng = np.random.default_rng(partida)
    melhor = None
    for _ in range(n_init):
        centros = [Z[rng.integers(len(Z))]]
        for _ in range(k - 1):
            d = _distancias(Z, np.array(centros)).min(axis=1)
            centros.append(Z[rng.choice(len(Z), p=d / d.sum())])
        centros = np.array(centros)
        for _ in range(300):
            grupo = _distancias(Z, centros).argmin(axis=1)
            novos = np.array([Z[grupo == j].mean(axis=0) if (grupo == j).any() else centros[j] for j in range(k)])
            if np.allclose(novos, centros):
                break
            centros = novos
        grupo = _distancias(Z, centros).argmin(axis=1)
        soma = float(((Z - centros[grupo]) ** 2).sum())
        if melhor is None or soma < melhor[0]:
            melhor = (soma, grupo, centros)
    return numerar(melhor[1], melhor[2])


def numerar(grupo, centros):
    """Grupo 0 é o maior, depois o segundo maior, e assim por diante (empate: centro menor)."""
    k = len(centros)
    tamanho = np.bincount(grupo, minlength=k)
    ordem = sorted(range(k), key=lambda j: (-tamanho[j], tuple(centros[j])))
    novo = np.empty(k, dtype=int)
    novo[ordem] = np.arange(k)
    return novo[grupo], centros[ordem]


def silhouette(Z, grupo):
    """Média do silhouette: (b - a) / max(a, b), com a = distância média ao próprio grupo."""
    k = int(grupo.max()) + 1
    if k < 2:
        raise ValueError('silhouette precisa de pelo menos 2 grupos')
    D = np.sqrt(np.maximum(_distancias(Z, Z), 0))
    n = np.bincount(grupo, minlength=k)
    soma = np.stack([D[:, grupo == j].sum(axis=1) for j in range(k)], axis=1)
    linhas = np.arange(len(Z))
    a = soma[linhas, grupo] / np.maximum(n[grupo] - 1, 1)
    media = soma / n
    media[linhas, grupo] = np.inf
    b = media.min(axis=1)
    s = (b - a) / np.maximum(a, b)
    s[n[grupo] == 1] = 0.0
    return float(s.mean())


def casar(a, b, k):
    """Para cada grupo de `a`, o grupo de `b` que mais se parece com ele (casamento de um para um)."""
    juntos = np.zeros((k, k), dtype=int)
    np.add.at(juntos, (a, b), 1)
    melhor = max(permutations(range(k)), key=lambda p: sum(juntos[i, p[i]] for i in range(k)))
    return np.array(melhor)


def trocam(a, b, k):
    """Quantas linhas mudam de grupo entre `a` e `b`, depois de casar os números dos grupos."""
    par = casar(a, b, k)
    return int((par[a] != b).sum())


def estabilidade(Z, k, rodadas=20, fracao=0.8, partida=0):
    """Refaz o K-means em `rodadas` amostras de `fracao` dos clientes, sem repetir cliente.

    Cada rodada põe todos os clientes no centro mais perto e compara com o arranjo da base
    inteira. Retorna (fração que troca em cada rodada, fração de cada grupo que fica junto em
    cada rodada) e o arranjo da base inteira.
    """
    base, _ = kmeans(Z, k, partida)
    rng = np.random.default_rng(partida + 1000)
    trocas, ficam = [], []
    for r in range(rodadas):
        amostra = rng.choice(len(Z), size=int(fracao * len(Z)), replace=False)
        _, centros = kmeans(Z[amostra], k, partida + 1 + r)
        g = _distancias(Z, centros).argmin(axis=1)
        par = casar(base, g, k)
        mesmo = par[base] == g
        trocas.append(1 - mesmo.mean())
        ficam.append([mesmo[base == j].mean() for j in range(k)])
    return np.array(trocas), np.array(ficam), base


def sortear_sem_email(grupo, fracao, partida):
    """Sorteia `fracao` de cada grupo para não receber e-mail. Retorna um vetor True/False."""
    if not 0 < fracao < 1:
        raise ValueError('a fração sorteada fica entre 0 e 1')
    rng = np.random.default_rng(partida)
    fora = np.zeros(len(grupo), dtype=bool)
    for j in np.unique(grupo):
        membros = np.flatnonzero(grupo == j)
        n = max(1, round(fracao * len(membros)))
        fora[rng.choice(membros, size=n, replace=False)] = True
    return fora


def cabe(tamanhos, emails, minimo):
    """O arranjo cabe na campanha: no máximo `emails` grupos, todos com pelo menos `minimo` clientes."""
    tamanhos = list(tamanhos)
    return len(tamanhos) <= emails and min(tamanhos) >= minimo
