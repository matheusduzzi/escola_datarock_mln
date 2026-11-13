"""Três formas de grupo e o que cada método faz com elas.

    python3 formas.py                         # as três formas, todos os métodos
    python3 formas.py --forma aneis           # só uma forma
    python3 formas.py --forma densidades --varrer
    python3 formas.py --forma ponte --raio 0.8

As formas:
- aneis: os dois anéis da aula «Kernel PCA, t-SNE e UMAP», 60 pontos cada, raios 1 e 3.
- densidades: uma mancha apertada de 100 pontos, uma espalhada de 60 ao lado e 10 pontos soltos.
- ponte: duas manchas de 70 pontos ligadas por uma fila de 12 pontos.

Aqui a gente sabe de que grupo cada ponto veio, porque as formas foram montadas assim. Isso serve
para medir os métodos. Nos clientes, ninguém sabe.
Os pontos saem iguais toda vez. Rode de dentro da pasta da aula.
"""
import argparse
import sys
import warnings

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import DBSCAN, HDBSCAN, KMeans

FORMAS = ['aneis', 'densidades', 'ponte']
NOMES = {'aneis': 'Dois anéis', 'densidades': 'Densidades diferentes', 'ponte': 'Duas manchas e uma ponte'}
# Raio do DBSCAN e tamanho mínimo do HDBSCAN que cada forma usa sem opção.
PADRAO = {'aneis': (1.0, 5), 'densidades': (0.7, 5), 'ponte': (0.5, 10)}
MINIMO = 5
RAIOS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2]


class Erro(Exception):
    pass


def br(x, casas=1):
    return f'{float(x):.{casas}f}'.replace('.', ',')


def numero(texto):
    try:
        return float(texto.replace(',', '.'))
    except ValueError:
        raise argparse.ArgumentTypeError(f'"{texto}" não é um número')


def aneis():
    rng = np.random.default_rng(4)
    angulos = np.linspace(0, 2 * np.pi, 60, endpoint=False)
    partes = [np.c_[r * np.cos(angulos), r * np.sin(angulos)] + rng.normal(0, 0.15, (60, 2)) for r in (1.0, 3.0)]
    return np.r_[partes[0], partes[1]], np.r_[np.zeros(60), np.ones(60)].astype(int)


def densidades():
    rng = np.random.default_rng(7)
    apertada = rng.normal([0, 0], 0.25, (100, 2))
    espalhada = rng.normal([2.6, 0], 1.0, (60, 2))
    soltos = rng.uniform([-3, -4], [7, 4], (10, 2))
    return np.r_[apertada, espalhada, soltos], np.r_[np.zeros(100), np.ones(60), -np.ones(10)].astype(int)


def ponte():
    rng = np.random.default_rng(5)
    a = rng.normal([0, 0], 0.5, (70, 2))
    b = rng.normal([6, 0], 0.5, (70, 2))
    fila = np.c_[np.linspace(1, 5, 12), rng.normal(0, 0.05, 12)]
    return np.r_[a, b, fila], np.r_[np.zeros(70), np.ones(70), -np.ones(12)].astype(int)


GERAR = {'aneis': aneis, 'densidades': densidades, 'ponte': ponte}


def medir(rotulos, verdade):
    """Compara os grupos achados com a origem dos pontos.

    Cada grupo achado vale pela origem da maioria dos pontos dele. Errado é o ponto de outra origem.
    Os pontos sem origem (soltos ou da ponte) não entram em certo nem errado.
    """
    rotulos = np.asarray(rotulos)
    tem = verdade >= 0
    certos = errados = 0
    for g in sorted(set(rotulos[tem]) - {-1}):
        nele = tem & (rotulos == g)
        maioria = np.bincount(verdade[nele]).max()
        certos += int(maioria)
        errados += int(nele.sum() - maioria)
    return {'grupos': len(set(rotulos) - {-1}), 'certos': certos, 'errados': errados,
            'ruido_grupo': int((tem & (rotulos == -1)).sum()),
            'soltos_ruido': int(((~tem) & (rotulos == -1)).sum()), 'soltos': int((~tem).sum())}


def metodos(x, k, raio, tamanho):
    return [
        ('K-means', f'K = {k}', KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(x)),
        ('Ward', f'{k} grupos', fcluster(linkage(x, 'ward'), k, 'maxclust')),
        ('menor distância', f'{k} grupos', fcluster(linkage(x, 'single'), k, 'maxclust')),
        ('DBSCAN', f'raio {br(raio, 2)}, mín. {MINIMO}', DBSCAN(eps=raio, min_samples=MINIMO).fit_predict(x)),
        ('HDBSCAN', f'tamanho mín. {tamanho}', HDBSCAN(min_cluster_size=tamanho, copy=True).fit_predict(x)),
    ]


def titulo(texto):
    print(texto)
    print('-' * len(texto))


def mostrar(forma, raio, tamanho):
    x, verdade = GERAR[forma]()
    k = len(set(verdade) - {-1})
    soltos = int((verdade < 0).sum())
    titulo(f'{NOMES[forma]}: {len(x)} pontos, {k} grupos de verdade' + (f' e {soltos} pontos sem grupo' if soltos else ''))
    print('método            parâmetro                grupos  certos  errados  ruído (de grupo)' + ('  sem grupo no ruído' if soltos else ''))
    for nome, parametro, rot in metodos(x, k, raio, tamanho):
        m = medir(rot, verdade)
        linha = f'{nome:<17} {parametro:<24} {m["grupos"]:>6} {m["certos"]:>7} {m["errados"]:>8} {m["ruido_grupo"]:>17}'
        if soltos:
            linha += f' {m["soltos_ruido"]:>12} de {m["soltos"]}'
        print(linha)
    print()


def mostrar_varrer(forma):
    x, verdade = GERAR[forma]()
    soltos = int((verdade < 0).sum())
    titulo(f'{NOMES[forma]}: DBSCAN com vários raios, mínimo de {MINIMO} pontos')
    print('  raio  grupos  certos  errados  ruído (de grupo)' + ('  sem grupo no ruído' if soltos else ''))
    for raio in RAIOS:
        m = medir(DBSCAN(eps=raio, min_samples=MINIMO).fit_predict(x), verdade)
        linha = f'{br(raio):>6} {m["grupos"]:>7} {m["certos"]:>7} {m["errados"]:>8} {m["ruido_grupo"]:>17}'
        if soltos:
            linha += f' {m["soltos_ruido"]:>12} de {m["soltos"]}'
        print(linha)
    print()


def main(argv):
    p = argparse.ArgumentParser(description='Três formas de grupo e cinco métodos.')
    p.add_argument('--forma', choices=FORMAS + ['todas'], default='todas')
    p.add_argument('--raio', type=numero, help='raio (eps) do DBSCAN; sem ela, o de cada forma')
    p.add_argument('--tamanho', type=int, help='min_cluster_size do HDBSCAN; sem ela, o de cada forma')
    p.add_argument('--varrer', action='store_true', help='DBSCAN com vários raios')
    a = p.parse_args(argv)
    try:
        if a.varrer and (a.raio is not None or a.tamanho is not None):
            raise Erro('--varrer já escolhe os raios; não use com --raio nem --tamanho')
        if a.raio is not None and not 0 < a.raio <= 10:
            raise Erro('--raio precisa ser maior que 0 e no máximo 10')
        if a.tamanho is not None and not 2 <= a.tamanho <= 100:
            raise Erro('--tamanho precisa ficar entre 2 e 100')
        for forma in (FORMAS if a.forma == 'todas' else [a.forma]):
            if a.varrer:
                mostrar_varrer(forma)
            else:
                raio, tamanho = PADRAO[forma]
                mostrar(forma, raio if a.raio is None else a.raio, tamanho if a.tamanho is None else a.tamanho)
    except Erro as e:
        print(f'erro: {e}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    warnings.simplefilter('error')
    sys.exit(main(sys.argv[1:]))
