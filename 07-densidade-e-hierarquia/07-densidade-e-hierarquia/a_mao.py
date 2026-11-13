"""DBSCAN, HDBSCAN e hierarquia em poucos clientes, passo a passo, para fazer a conta à mão.

    python3 a_mao.py                              # os 12 clientes e a tabela de distâncias
    python3 a_mao.py --dbscan 3 --min 3           # DBSCAN: vizinhos, núcleo, borda e ruído
    python3 a_mao.py --varrer --min 3             # DBSCAN com vários raios
    python3 a_mao.py --hdbscan 3                  # HDBSCAN: distância de núcleo e grupos
    python3 a_mao.py --hierarquia single          # as junções do dendrograma, da menor altura à maior
    python3 a_mao.py --hierarquia ward --cortar 3

Duas colunas: pedidos no ano e ticket médio em centenas de reais, com uma casa.
Rode de dentro da pasta da aula.
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN, HDBSCAN

ARQUIVO = Path('..') / 'dados' / 'clientes' / 'clientes.csv'
PADRAO = [27, 78, 104, 1, 15, 23, 105, 137, 63, 98, 143, 77]
METODOS = {'single': 'menor distância (single)', 'complete': 'maior distância (complete)',
           'average': 'distância média (average)', 'ward': 'Ward'}
RAIOS = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 5, 10, 12, 16, 35]
LETRAS = 'ABCDEFGHIJKLMNOPQRST'


class Erro(Exception):
    pass


def br(x, casas=1):
    return f'{float(x):.{casas}f}'.replace('.', ',')


def numero(texto):
    try:
        return float(texto.replace(',', '.'))
    except ValueError:
        raise argparse.ArgumentTypeError(f'"{texto}" não é um número')


def lista_ids(texto):
    try:
        ids = [int(p) for p in texto.split(',') if p.strip()]
    except ValueError:
        raise argparse.ArgumentTypeError(f'"{texto}" precisa ser uma lista de ids separados por vírgula')
    if not ids:
        raise argparse.ArgumentTypeError('a lista de ids está vazia')
    return ids


def ler_argumentos(argv):
    p = argparse.ArgumentParser(description='DBSCAN, HDBSCAN e hierarquia em poucos clientes.')
    modo = p.add_mutually_exclusive_group()
    modo.add_argument('--dbscan', type=numero, metavar='RAIO', help='roda o DBSCAN com esse raio (eps)')
    modo.add_argument('--varrer', action='store_true', help='roda o DBSCAN com vários raios')
    modo.add_argument('--hdbscan', type=int, metavar='N', help='roda o HDBSCAN com min_cluster_size = N')
    modo.add_argument('--hierarquia', choices=list(METODOS), help='mostra as junções do agrupamento hierárquico')
    p.add_argument('--min', type=int, default=None, help='min_samples do DBSCAN ou do HDBSCAN (conta o próprio cliente)')
    p.add_argument('--cortar', type=int, metavar='K', help='com --hierarquia: corta a árvore em K grupos')
    p.add_argument('--altura', type=numero, metavar='H', help='com --hierarquia: corta a árvore nessa altura')
    p.add_argument('--ids', type=lista_ids, help='outros clientes no lugar dos 12 da aula (de 4 a 20)')
    p.add_argument('--sem', type=lista_ids, help='tira esses clientes da conta')
    return p.parse_args(argv)


def validar(a, base):
    ids = list(a.ids or PADRAO)
    if len(set(ids)) != len(ids):
        raise Erro('tem cliente repetido em --ids')
    faltam = [i for i in ids if i not in base.index]
    if faltam:
        raise Erro(f'esses clientes não existem na base: {faltam}')
    if a.sem:
        fora = [i for i in a.sem if i not in ids]
        if fora:
            raise Erro(f'--sem tem clientes que não estão na lista: {fora}')
        ids = [i for i in ids if i not in a.sem]
    if not 4 <= len(ids) <= 20:
        raise Erro(f'a conta precisa de 4 a 20 clientes; ficaram {len(ids)}')
    if a.min is not None and not (a.dbscan is not None or a.varrer or a.hdbscan is not None):
        raise Erro('--min só vale com --dbscan, --varrer ou --hdbscan')
    if a.min is not None and not 1 <= a.min <= len(ids):
        raise Erro(f'--min precisa ficar entre 1 e {len(ids)}')
    if a.dbscan is not None and not 0 < a.dbscan <= 100:
        raise Erro('o raio do --dbscan precisa ser maior que 0 e no máximo 100')
    if a.hdbscan is not None and not 2 <= a.hdbscan <= len(ids):
        raise Erro(f'--hdbscan precisa ficar entre 2 e {len(ids)}')
    if (a.cortar is not None or a.altura is not None) and not a.hierarquia:
        raise Erro('--cortar e --altura só valem com --hierarquia')
    if a.cortar is not None and a.altura is not None:
        raise Erro('use --cortar ou --altura, não os dois')
    if a.cortar is not None and not 1 <= a.cortar <= len(ids):
        raise Erro(f'--cortar precisa ficar entre 1 e {len(ids)}')
    if a.altura is not None and a.altura <= 0:
        raise Erro('--altura precisa ser maior que 0')
    return ids


def ler_clientes():
    if not ARQUIVO.is_file():
        raise Erro(f'não achei {ARQUIVO}. Rode de dentro da pasta da aula, 07-densidade-e-hierarquia.')
    return pd.read_csv(ARQUIVO).set_index('id_cliente')


def tabela(base, ids):
    x = np.c_[base.loc[ids, 'pedidos_12m'].to_numpy(float), np.round(base.loc[ids, 'ticket_medio'].to_numpy(float) / 100, 1)]
    return x


def titulo(texto):
    print(texto)
    print('-' * len(texto))


def mostrar_clientes(ids, x):
    print('cliente      pedidos  ticket (R$ 100)')
    for i, (p, t) in zip(ids, x):
        print(f'{i:>7} {int(p):>12} {br(t):>16}')


def mostrar_distancias(ids, d):
    print()
    titulo('Distância entre cada par de clientes')
    print('cliente' + ''.join(f'{i:>7}' for i in ids))
    for i, linha in zip(ids, d):
        print(f'{i:>7}' + ''.join(f'{br(v):>7}' for v in linha))


def nomes(ids, membros):
    return ', '.join(str(ids[m]) for m in membros)


def grupos_texto(ids, rotulos):
    partes = []
    for g in sorted(set(rotulos) - {-1}):
        partes.append(f'{LETRAS[g]}: {nomes(ids, np.flatnonzero(rotulos == g))}')
    texto = '; '.join(partes) if partes else 'nenhum grupo'
    ruido = np.flatnonzero(rotulos == -1)
    return texto + (f'; ruído: {nomes(ids, ruido)}' if len(ruido) else '; ruído: ninguém')


def dbscan_a_mao(d, raio, minimo):
    """O DBSCAN escrito passo a passo. Retorna (vizinhos, núcleo, rótulos), na ordem da lista."""
    n = len(d)
    vizinhos = [np.flatnonzero(d[i] <= raio) for i in range(n)]
    nucleo = np.array([len(v) >= minimo for v in vizinhos])
    rotulos = np.full(n, -1)
    grupo = 0
    for i in range(n):
        if not nucleo[i] or rotulos[i] != -1:
            continue
        rotulos[i] = grupo
        fila = [i]
        while fila:
            atual = fila.pop(0)
            if not nucleo[atual]:
                continue
            for v in vizinhos[atual]:
                if rotulos[v] == -1:
                    rotulos[v] = grupo
                    fila.append(v)
        grupo += 1
    return vizinhos, nucleo, rotulos


def mostrar_dbscan(ids, x, d, raio, minimo):
    vizinhos, nucleo, rotulos = dbscan_a_mao(d, raio, minimo)
    titulo(f'DBSCAN com raio {br(raio, 2)} e mínimo de {minimo} clientes na vizinhança (contando ele mesmo)')
    print('cliente      pedidos  ticket (R$ 100)  vizinhos no raio   quantos  tipo    grupo')
    for k, i in enumerate(ids):
        outros = [ids[v] for v in vizinhos[k] if v != k]
        if nucleo[k]:
            tipo = 'núcleo'
        elif rotulos[k] != -1:
            tipo = 'borda'
        else:
            tipo = 'ruído'
        grupo = LETRAS[rotulos[k]] if rotulos[k] != -1 else '-'
        lista = ', '.join(map(str, outros)) or 'nenhum'
        print(f'{i:>7} {int(x[k, 0]):>12} {br(x[k, 1]):>16}  {lista:<17} {len(vizinhos[k]):>7}  {tipo:<7} {grupo}')
    print()
    print('núcleo: tem pelo menos o mínimo de clientes no raio. borda: não tem, mas está no raio de um núcleo.')
    print('ruído: nem uma coisa nem outra.')
    print(f'grupos: {grupos_texto(ids, rotulos)}')
    sk = DBSCAN(eps=raio, min_samples=minimo).fit_predict(x)
    igual = np.array_equal(sk, rotulos)
    print(f'o DBSCAN do scikit-learn, com eps={br(raio, 2)} e min_samples={minimo}, deu os mesmos grupos: {"sim" if igual else "NÃO"}')
    if not igual:
        raise Erro('o DBSCAN passo a passo e o do scikit-learn deram grupos diferentes')


def mostrar_varrer(ids, x, d, minimo):
    titulo(f'DBSCAN com vários raios e mínimo de {minimo} clientes na vizinhança')
    print('   raio  grupos  ruído  grupos e ruído')
    for raio in RAIOS:
        _, _, rotulos = dbscan_a_mao(d, raio, minimo)
        sk = DBSCAN(eps=raio, min_samples=minimo).fit_predict(x)
        if not np.array_equal(sk, rotulos):
            raise Erro(f'o DBSCAN passo a passo e o do scikit-learn deram grupos diferentes no raio {br(raio)}')
        n = len(set(rotulos) - {-1})
        print(f'{br(raio):>7} {n:>7} {int((rotulos == -1).sum()):>6}  {grupos_texto(ids, rotulos)}')


def mostrar_hdbscan(ids, x, d, tamanho, minimo):
    m = tamanho if minimo is None else minimo
    nucleo = np.sort(d, axis=1)[:, m - 1]
    titulo(f'HDBSCAN com min_cluster_size={tamanho} e min_samples={m}')
    print(f'Distância de núcleo: a distância até o {m}º cliente mais perto, contando ele mesmo.')
    print('cliente      pedidos  ticket (R$ 100)  dist. de núcleo')
    for k, i in enumerate(ids):
        print(f'{i:>7} {int(x[k, 0]):>12} {br(x[k, 1]):>16} {br(nucleo[k], 2):>16}')
    alcance = np.maximum(d, np.maximum(nucleo[:, None], nucleo[None, :]))
    np.fill_diagonal(alcance, 0)
    print()
    titulo('Junções pela distância de alcance: a maior entre a distância e as duas distâncias de núcleo')
    mostrar_juncoes(ids, linkage(squareform(alcance, checks=False), 'single'))
    h = HDBSCAN(min_cluster_size=tamanho, min_samples=minimo, copy=True).fit(x)
    print()
    print(f'grupos do HDBSCAN: {grupos_texto(ids, em_ordem(h.labels_))}')
    print('força de cada cliente no grupo (probabilities_, de 0 a 1; 0 = ruído):')
    print('  ' + '; '.join(f'{i} {br(p, 2)}' for i, p in zip(ids, h.probabilities_)))


def em_ordem(rotulos):
    """Renumera os grupos na ordem em que aparecem na lista; o ruído continua -1."""
    mapa = {}
    for r in rotulos:
        if r != -1:
            mapa.setdefault(r, len(mapa))
    return np.array([mapa.get(r, -1) for r in rotulos])


def mostrar_juncoes(ids, L):
    membros = {k: [k] for k in range(len(ids))}
    print('passo  junta                                            com                                              altura  tamanho')
    for passo, (a, b, altura, tam) in enumerate(L, start=1):
        a, b = int(a), int(b)
        ga, gb = sorted(membros[a]), sorted(membros[b])
        print(f'{passo:>5}  {nomes(ids, ga):<48} {nomes(ids, gb):<48} {br(altura, 2):>6} {int(tam):>8}')
        membros[len(ids) + passo - 1] = ga + gb


def mostrar_hierarquia(ids, x, metodo, cortar, altura):
    L = linkage(x, metodo)
    titulo(f'Agrupamento hierárquico, ligação por {METODOS[metodo]}')
    mostrar_juncoes(ids, L)
    saltos = np.diff(L[:, 2])
    print()
    print('os três maiores saltos de altura entre um passo e o seguinte:')
    for k in sorted(np.argsort(-saltos, kind='stable')[:3]):
        print(f'  do passo {k + 1} ({br(L[k, 2], 2)}) para o {k + 2} ({br(L[k + 1, 2], 2)}): salto de '
              f'{br(saltos[k], 2)}; cortar aí deixa {len(ids) - k - 1} grupo(s)')
    if cortar is not None or altura is not None:
        if cortar is not None:
            rot = fcluster(L, cortar, 'maxclust')
            texto = f'em {cortar} grupo(s)'
        else:
            rot = fcluster(L, altura, 'distance')
            texto = f'na altura {br(altura, 2)}'
        print(f'cortando {texto}: {grupos_texto(ids, em_ordem(rot))}')


def main(argv):
    a = ler_argumentos(argv)
    try:
        base = ler_clientes()
        ids = validar(a, base)
        x = tabela(base, ids)
        d = squareform(pdist(x))
        if a.dbscan is not None:
            mostrar_dbscan(ids, x, d, a.dbscan, 3 if a.min is None else a.min)
        elif a.varrer:
            mostrar_varrer(ids, x, d, 3 if a.min is None else a.min)
        elif a.hdbscan is not None:
            mostrar_hdbscan(ids, x, d, a.hdbscan, a.min)
        elif a.hierarquia:
            mostrar_hierarquia(ids, x, a.hierarquia, a.cortar, a.altura)
        else:
            mostrar_clientes(ids, x)
            mostrar_distancias(ids, d)
    except Erro as e:
        print(f'erro: {e}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    warnings.simplefilter('error')
    sys.exit(main(sys.argv[1:]))
