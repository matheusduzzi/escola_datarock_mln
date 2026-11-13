"""DBSCAN, HDBSCAN e hierarquia nos 3000 clientes da Loja Exemplo.

    python3 base.py --vizinho 8                # a distância até o 8º vizinho: ajuda a escolher o raio
    python3 base.py --varrer --min 8           # DBSCAN com vários raios
    python3 base.py --dbscan 1.5 --min 8       # os grupos e o ruído de um DBSCAN
    python3 base.py --hdbscan 50               # HDBSCAN com min_cluster_size = 50
    python3 base.py --hierarquia ward          # os níveis do dendrograma, de 1 a 10 grupos
    python3 base.py --hierarquia ward --k 5    # agrupamento hierárquico cortado em 5 grupos
    python3 base.py --hdbscan 50 --salvar grupos.csv

As oito colunas de comportamento entram na escala z, como na aula «PCA».
Rode de dentro da pasta da aula.
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import DBSCAN, HDBSCAN
from sklearn.neighbors import NearestNeighbors

ARQUIVO = Path('..') / 'dados' / 'clientes' / 'clientes.csv'
COLUNAS = ['pedidos_12m', 'gasto_12m', 'ticket_medio', 'dias_desde_ultima_compra',
           'categorias_distintas', 'pct_pedidos_cupom', 'pct_pedidos_app', 'devolucoes_12m']
CURTOS = ['pedidos', 'gasto R$', 'ticket R$', 'dias', 'categ.', 'cupom %', 'app %', 'devol.']
METODOS = {'single': 'menor distância (single)', 'complete': 'maior distância (complete)',
           'average': 'distância média (average)', 'ward': 'Ward'}
TOPO = 40
RAIOS = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
QUANTIS = [50, 75, 90, 95, 98, 99]


class Erro(Exception):
    pass


def br(x, casas=1):
    texto = f'{float(x):.{casas}f}'.replace('.', ',')
    return texto[1:] if texto.startswith('-') and set(texto[1:]) <= set('0,') else texto


def milhar(n):
    return f'{int(round(n)):,}'.replace(',', '.')


def numero(texto):
    try:
        return float(texto.replace(',', '.'))
    except ValueError:
        raise argparse.ArgumentTypeError(f'"{texto}" não é um número')


def ler_argumentos(argv):
    p = argparse.ArgumentParser(description='DBSCAN, HDBSCAN e hierarquia nos clientes da Loja Exemplo.')
    modo = p.add_mutually_exclusive_group(required=True)
    modo.add_argument('--vizinho', type=int, metavar='N', help='distância de cada cliente até o N-ésimo vizinho')
    modo.add_argument('--varrer', action='store_true', help='DBSCAN com vários raios')
    modo.add_argument('--dbscan', type=numero, metavar='RAIO', help='DBSCAN com esse raio (eps)')
    modo.add_argument('--hdbscan', type=int, metavar='N', help='HDBSCAN com min_cluster_size = N')
    modo.add_argument('--hierarquia', choices=list(METODOS), help='agrupamento hierárquico')
    p.add_argument('--min', type=int, help='min_samples do DBSCAN (padrão 8) ou do HDBSCAN (padrão: o N do --hdbscan)')
    p.add_argument('--k', type=int, help='com --hierarquia: em quantos grupos cortar (de 2 a 10); sem ela, os níveis')
    p.add_argument('--ruido-por-ward', type=int, metavar='K',
                   help='com --dbscan ou --hdbscan: quantos clientes de cada grupo do Ward com K grupos foram para o ruído')
    p.add_argument('--sem-topo', action='store_true', help=f'tira os clientes de {TOPO} ou mais pedidos')
    p.add_argument('--salvar', metavar='ARQUIVO.csv', help='grava id_cliente e grupo (-1 = ruído)')
    return p.parse_args(argv)


def validar(a):
    if a.vizinho is not None and not 2 <= a.vizinho <= 100:
        raise Erro('--vizinho precisa ficar entre 2 e 100')
    if a.min is not None and (a.vizinho is not None or a.hierarquia):
        raise Erro('--min só vale com --varrer, --dbscan ou --hdbscan')
    if a.min is not None and not 1 <= a.min <= 100:
        raise Erro('--min precisa ficar entre 1 e 100')
    if a.dbscan is not None and not 0 < a.dbscan <= 10:
        raise Erro('o raio do --dbscan precisa ser maior que 0 e no máximo 10')
    if a.hdbscan is not None and not 2 <= a.hdbscan <= 500:
        raise Erro('--hdbscan precisa ficar entre 2 e 500')
    if a.k is not None and not a.hierarquia:
        raise Erro('--k só vale com --hierarquia')
    if a.k is not None and not 2 <= a.k <= 10:
        raise Erro('--k precisa ficar entre 2 e 10')
    if a.salvar and (a.vizinho is not None or a.varrer or (a.hierarquia and a.k is None)):
        raise Erro('--salvar só vale com --dbscan, --hdbscan ou --hierarquia com --k')
    if a.ruido_por_ward is not None and a.dbscan is None and a.hdbscan is None:
        raise Erro('--ruido-por-ward só vale com --dbscan ou --hdbscan')
    if a.ruido_por_ward is not None and not 2 <= a.ruido_por_ward <= 10:
        raise Erro('--ruido-por-ward precisa ficar entre 2 e 10')
    if a.salvar and not a.salvar.endswith('.csv'):
        raise Erro('--salvar precisa de um nome que termine em .csv')


def ler_clientes():
    if not ARQUIVO.is_file():
        raise Erro(f'não achei {ARQUIVO}. Rode de dentro da pasta da aula, 07-densidade-e-hierarquia.')
    return pd.read_csv(ARQUIVO)


def escala_z(base):
    x = base[COLUNAS].to_numpy(float)
    return (x - x.mean(axis=0)) / x.std(axis=0, ddof=1)


def titulo(texto):
    print(texto)
    print('-' * len(texto))


def mostrar_vizinho(z, n):
    d, _ = NearestNeighbors(n_neighbors=n).fit(z).kneighbors(z)
    k = d[:, -1]
    titulo(f'Distância de cada cliente até o {n}º cliente mais perto, contando ele mesmo')
    print('   quantil  distância')
    for q in QUANTIS:
        print(f'{q:>9}% {br(np.percentile(k, q), 2):>10}')
    print(f'{"maior":>10} {br(k.max(), 2):>10}')
    print()
    print('clientes com essa distância acima de cada raio (no DBSCAN com min_samples igual, eles não são núcleo):')
    print('   raio  clientes')
    for r in RAIOS:
        print(f'{br(r, 2):>7} {int((k > r).sum()):>9}')


def tabela_grupos(base, rotulos, extra=''):
    topo = base['pedidos_12m'].to_numpy() >= TOPO
    print('grupo  clientes  do topo' + ''.join(f'{c:>11}' for c in CURTOS) + '   <- média do grupo' + extra)
    for g in sorted(set(rotulos), key=lambda g: (g == -1, g)):
        nele = rotulos == g
        media = base.loc[nele, COLUNAS].mean().to_numpy()
        valores = [br(media[0]), milhar(media[1]), milhar(media[2]), milhar(media[3]), br(media[4]),
                   milhar(media[5]), milhar(media[6]), br(media[7])]
        nome = 'ruído' if g == -1 else str(g + 1)
        print(f'{nome:>5} {int(nele.sum()):>9} {int((nele & topo).sum()):>8}' + ''.join(f'{v:>11}' for v in valores))


def resumo(base, rotulos):
    topo = base['pedidos_12m'].to_numpy() >= TOPO
    tamanhos = sorted(np.bincount(rotulos[rotulos >= 0]), reverse=True) if (rotulos >= 0).any() else []
    return (len(tamanhos), int((rotulos == -1).sum()), int(tamanhos[0]) if tamanhos else 0,
            int(((rotulos == -1) & topo).sum()))


def mostrar_varrer(base, z, minimo):
    titulo(f'DBSCAN com vários raios, min_samples={minimo}, {len(base)} clientes na escala z')
    print('   raio  grupos  ruído  maior grupo  do topo no ruído')
    for r in RAIOS:
        n, ruido, maior, topo = resumo(base, DBSCAN(eps=r, min_samples=minimo).fit_predict(z))
        print(f'{br(r, 2):>7} {n:>7} {ruido:>6} {maior:>12} {topo:>17}')


def mostrar_niveis(base, z, metodo, sem):
    L = linkage(z, metodo)
    topo = base['pedidos_12m'].to_numpy() >= TOPO
    titulo(f'Níveis do dendrograma, ligação por {METODOS[metodo]}, {len(base)} clientes{sem}')
    print('Cortar a árvore numa altura dá um número de grupos. Do 1 ao 10:')
    print(' grupos  altura do corte  tamanhos dos grupos (o do topo entre colchetes)')
    for k in range(1, 11):
        rot = fcluster(L, k, 'maxclust')
        tamanhos = []
        for g in sorted(set(rot), key=lambda g: -(rot == g).sum()):
            n = int((rot == g).sum())
            tamanhos.append(f'[{n}]' if (rot == g)[topo].all() and n == int(topo.sum()) else str(n))
        altura = br(L[-k, 2], 2) if k > 1 else br(L[-1, 2], 2) + ' (acima)'
        print(f'{k:>7}  {altura:>15}  {", ".join(tamanhos)}')
    print()
    print('Cada linha divide um grupo da linha de cima em dois. A altura é a do passo que a linha desfaz.')


def mostrar_ruido_por_ward(base, z, rotulos, k):
    ward = fcluster(linkage(z, 'ward'), k, 'maxclust') - 1
    print()
    titulo(f'O ruído em cada grupo do Ward com {k} grupos')
    print('grupo do Ward  clientes  no ruído  ticket R$')
    for g in range(k):
        nele = ward == g
        print(f'{g + 1:>13} {int(nele.sum()):>9} {int((nele & (rotulos == -1)).sum()):>9} '
              f'{milhar(base.loc[nele, "ticket_medio"].mean()):>10}')


def salvar(base, rotulos, arquivo):
    pd.DataFrame({'id_cliente': base['id_cliente'], 'grupo': np.where(rotulos == -1, -1, rotulos + 1)}).to_csv(
        arquivo, index=False)
    print(f'gravei {arquivo}: {len(base)} clientes, grupo -1 = ruído')


def main(argv):
    a = ler_argumentos(argv)
    try:
        validar(a)
        base = ler_clientes()
        if a.sem_topo:
            base = base[base['pedidos_12m'] < TOPO].reset_index(drop=True)
        z = escala_z(base)
        sem = ' (sem os clientes do topo)' if a.sem_topo else ''
        if a.vizinho is not None:
            mostrar_vizinho(z, a.vizinho)
            return 0
        if a.varrer:
            mostrar_varrer(base, z, 8 if a.min is None else a.min)
            return 0
        if a.dbscan is not None:
            m = 8 if a.min is None else a.min
            rotulos = DBSCAN(eps=a.dbscan, min_samples=m).fit_predict(z)
            titulo(f'DBSCAN, raio {br(a.dbscan, 2)}, min_samples={m}, {len(base)} clientes{sem}')
        elif a.hdbscan is not None:
            rotulos = HDBSCAN(min_cluster_size=a.hdbscan, min_samples=a.min, copy=True).fit_predict(z)
            m = a.hdbscan if a.min is None else a.min
            titulo(f'HDBSCAN, min_cluster_size={a.hdbscan}, min_samples={m}, {len(base)} clientes{sem}')
        elif a.k is None:
            mostrar_niveis(base, z, a.hierarquia, sem)
            return 0
        else:
            rotulos = fcluster(linkage(z, a.hierarquia), a.k, 'maxclust') - 1
            titulo(f'Agrupamento hierárquico, ligação por {METODOS[a.hierarquia]}, cortado em {a.k} grupos, '
                   f'{len(base)} clientes{sem}')
        n, ruido, maior, topo = resumo(base, rotulos)
        print(f'{n} grupo(s); ruído: {ruido} cliente(s) ({br(100 * ruido / len(base))}%), {topo} deles do topo')
        print('a coluna "do topo" conta os clientes de 40 ou mais pedidos no ano')
        tabela_grupos(base, rotulos)
        if a.ruido_por_ward is not None:
            mostrar_ruido_por_ward(base, z, rotulos, a.ruido_por_ward)
        if a.salvar:
            salvar(base, rotulos, a.salvar)
    except Erro as e:
        print(f'erro: {e}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    warnings.simplefilter('error')
    sys.exit(main(sys.argv[1:]))
