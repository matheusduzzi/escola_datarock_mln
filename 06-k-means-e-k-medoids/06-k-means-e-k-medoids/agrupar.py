"""K-means e K-medoids nos 3000 clientes da Loja Exemplo, só com o NumPy e o pandas.

    python3 agrupar.py                         # K-means com K = 3, 10 inícios
    python3 agrupar.py --metodo kmedoids       # K-medoids com K = 3
    python3 agrupar.py --curva                 # a soma dos quadrados do K-means para K de 1 a 10
    python3 agrupar.py --repetir               # 20 rodadas de um início só: todas chegam no mesmo lugar?
    python3 agrupar.py --sem-topo              # sem os 30 clientes de 40 ou mais pedidos
    python3 agrupar.py --k 5 --salvar grupos.csv

As oito colunas de comportamento entram na escala z, como na aula «PCA».
Rode de dentro da pasta da aula.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ARQUIVO = Path('..') / 'dados' / 'clientes' / 'clientes.csv'
COLUNAS = ['pedidos_12m', 'gasto_12m', 'ticket_medio', 'dias_desde_ultima_compra',
           'categorias_distintas', 'pct_pedidos_cupom', 'pct_pedidos_app', 'devolucoes_12m']
CURTOS = ['pedidos', 'gasto R$', 'ticket R$', 'dias', 'categ.', 'cupom %', 'app %', 'devol.']
NOMES = {'kmeans': 'K-means', 'kmedoids': 'K-medoids'}
TOPO = 40             # clientes com 40 ou mais pedidos no ano
MAX_RODADAS = 300     # como o max_iter do KMeans do scikit-learn
REPETICOES = 20


class Erro(Exception):
    pass


def br(x, casas=1):
    texto = f'{float(x):.{casas}f}'.replace('.', ',')
    return texto[1:] if texto.startswith('-') and set(texto[1:]) <= set('0,') else texto


def milhar(n):
    return f'{int(round(n)):,}'.replace(',', '.')


def ler_argumentos(argv):
    p = argparse.ArgumentParser(description='K-means e K-medoids nos clientes da Loja Exemplo.')
    p.add_argument('--metodo', choices=['kmeans', 'kmedoids'], default='kmeans')
    p.add_argument('--k', type=int, default=3, help='quantos grupos (de 1 a 10; padrão 3)')
    p.add_argument('--inicios', type=int, default=10, help='quantos inícios sorteados; fica o de menor soma (padrão 10)')
    p.add_argument('--sorteio', type=int, default=1, help='sorteio dos inícios (padrão 1)')
    p.add_argument('--curva', action='store_true', help='K-means com K de 1 a 10 e a queda da soma a cada K')
    p.add_argument('--repetir', action='store_true', help=f'{REPETICOES} rodadas de um início só, sorteios 1 a {REPETICOES}')
    p.add_argument('--sem-topo', action='store_true', help=f'tira os clientes de {TOPO} ou mais pedidos')
    p.add_argument('--salvar', metavar='ARQUIVO.csv', help='grava id_cliente e grupo')
    return p.parse_args(argv)


def validar(a):
    if not 1 <= a.k <= 10:
        raise Erro('--k precisa ficar entre 1 e 10')
    if not 1 <= a.inicios <= 50:
        raise Erro('--inicios precisa ficar entre 1 e 50')
    if a.curva and (a.metodo != 'kmeans' or a.repetir or a.salvar):
        raise Erro('--curva roda só o K-means e não se combina com --repetir nem --salvar')
    if a.repetir and a.salvar:
        raise Erro('--repetir não se combina com --salvar')
    if a.salvar and not a.salvar.endswith('.csv'):
        raise Erro('--salvar precisa de um nome que termine em .csv')


def ler_clientes():
    if not ARQUIVO.is_file():
        raise Erro(f'não achei {ARQUIVO}. Rode de dentro da pasta da aula, 06-k-means-e-k-medoids.')
    return pd.read_csv(ARQUIVO)


def distancias2(a, b):
    d = (a * a).sum(axis=1)[:, None] + (b * b).sum(axis=1)[None, :] - 2 * np.dot(a, b.T)
    return np.maximum(d, 0)


# ---------- K-means ----------

def kmeans_mais_mais(z, k, rng):
    """Início k-means++: o primeiro centro é sorteado; cada próximo é sorteado com chance proporcional
    à distância ao quadrado até o centro mais perto já escolhido."""
    centros = [int(rng.integers(len(z)))]
    d2 = distancias2(z, z[centros]).min(axis=1)
    for _ in range(1, k):
        centros.append(int(rng.choice(len(z), p=d2 / d2.sum())))
        d2 = np.minimum(d2, distancias2(z, z[centros[-1:]]).ravel())
    return z[centros].copy()


def kmeans(z, k, rng):
    centros = kmeans_mais_mais(z, k, rng)
    grupos = None
    for rodada in range(1, MAX_RODADAS + 1):
        novos = distancias2(z, centros).argmin(axis=1)
        if grupos is not None and (novos == grupos).all():
            break
        grupos = novos
        for j in range(k):
            if (grupos == j).any():
                centros[j] = z[grupos == j].mean(axis=0)
    soma = float(distancias2(z, centros)[np.arange(len(z)), grupos].sum())
    return grupos, centros, soma, rodada


def melhor_kmeans(z, k, inicios, sorteio):
    rng = np.random.default_rng(sorteio)
    melhor = None
    for _ in range(inicios):
        r = kmeans(z, k, rng)
        if melhor is None or r[2] < melhor[2] - 1e-9:
            melhor = r
    return melhor


# ---------- K-medoids ----------

def kmedoids(d, k, rng):
    """Alternado, com início k-medoids++: cada cliente vai para o medoide mais perto; em cada grupo, o
    medoide vira o cliente com a menor soma de distâncias até os outros do grupo."""
    medoides = [int(rng.integers(len(d)))]
    perto = d[:, medoides[0]].copy()
    for _ in range(1, k):
        medoides.append(int(rng.choice(len(d), p=perto ** 2 / (perto ** 2).sum())))
        perto = np.minimum(perto, d[:, medoides[-1]])
    for rodada in range(1, MAX_RODADAS + 1):
        grupos = d[:, medoides].argmin(axis=1)
        novos = []
        for j, m in enumerate(medoides):
            membros = np.flatnonzero(grupos == j)
            s = d[np.ix_(membros, membros)].sum(axis=1)
            novos.append(m if np.isclose(s[membros == m].min(), s.min()) else int(membros[s.argmin()]))
        if novos == medoides:
            break
        medoides = novos
    grupos = d[:, medoides].argmin(axis=1)
    soma = float(d[np.arange(len(d)), np.array(medoides)[grupos]].sum())
    return grupos, medoides, soma, rodada


def melhor_kmedoids(d, k, inicios, sorteio):
    rng = np.random.default_rng(sorteio)
    melhor = None
    for _ in range(inicios):
        r = kmedoids(d, k, rng)
        if melhor is None or r[2] < melhor[2] - 1e-9:
            melhor = r
    return melhor


# ---------- saídas ----------

def ordem_por_pedidos(centros_originais):
    return list(np.argsort(-centros_originais[:, 0], kind='stable'))


def mostrar_grupos(ids, x, topo, grupos, centros_originais, medoides=None):
    ordem = ordem_por_pedidos(centros_originais)
    cab = '  '.join(f'{c:>9}' for c in CURTOS)
    centro = 'medoide' if medoides is not None else 'centro (média)'
    print(f'grupo  clientes  do topo  {cab}   <- {centro}')
    for nome, j in enumerate(ordem, 1):
        n = int((grupos == j).sum())
        c = centros_originais[j]
        valores = [br(c[0], 1), milhar(c[1]), milhar(c[2]), br(c[3], 0), br(c[4], 1), br(c[5], 0), br(c[6], 0), br(c[7], 1)]
        extra = f'   cliente {ids[medoides[j]]}' if medoides is not None else ''
        print(f'{nome:>5}  {n:>8}  {int(topo[grupos == j].sum()):>7}  ' + '  '.join(f'{v:>9}' for v in valores) + extra)
    return ordem


def main(argv):
    a = ler_argumentos(argv)
    validar(a)
    clientes = ler_clientes()
    x_todos = clientes[COLUNAS].to_numpy(float)
    z_todos = (x_todos - x_todos.mean(axis=0)) / x_todos.std(axis=0, ddof=1)
    topo_todos = x_todos[:, 0] >= TOPO
    manter = ~topo_todos if a.sem_topo else np.ones(len(x_todos), bool)
    x, z, topo = x_todos[manter], z_todos[manter], topo_todos[manter]
    ids = clientes['id_cliente'].to_numpy()[manter]
    print(f'clientes: {len(x)}{" (sem os " + str(topo_todos.sum()) + " de " + str(TOPO) + " ou mais pedidos)" if a.sem_topo else ""}; '
          f'{len(COLUNAS)} colunas na escala z dos 3000 clientes')
    print(f'clientes de {TOPO} ou mais pedidos ("do topo"): {int(topo.sum())}')
    print()

    if a.curva:
        print(f'K-means, {a.inicios} inícios k-means++ por K (sorteio {a.sorteio}); fica o de menor soma')
        print('K   soma dos quadrados    queda   menor grupo' + ('' if a.sem_topo else '   grupo com mais clientes do topo'))
        anterior = None
        for k in range(1, 11):
            grupos, _, soma, _ = melhor_kmeans(z, k, a.inicios, a.sorteio)
            tamanhos = np.bincount(grupos, minlength=k)
            j = int(np.bincount(grupos, weights=topo, minlength=k).argmax())
            queda = '' if anterior is None else br(anterior - soma, 0)
            extra = '' if a.sem_topo else f'   {int(topo[grupos == j].sum())} do topo num grupo de {int(tamanhos[j])}'
            print(f'{k:<2}  {br(soma, 0):>18}  {queda:>7}  {int(tamanhos.min()):>12}{extra}')
            anterior = soma
        print()
        print('soma dos quadrados: a soma, para cada cliente, da distância ao quadrado até o centro do grupo dele')
        return

    if a.metodo == 'kmedoids':
        d = np.sqrt(distancias2(z, z))
        np.fill_diagonal(d, 0)

    if a.repetir:
        print(f'{NOMES[a.metodo]} com K = {a.k}, {REPETICOES} rodadas de um início só (sorteios 1 a {REPETICOES})')
        somas = []
        for s in range(1, REPETICOES + 1):
            rng = np.random.default_rng(s)
            r = kmeans(z, a.k, rng) if a.metodo == 'kmeans' else kmedoids(d, a.k, rng)
            somas.append(round(r[2], 1))
        valores = sorted(set(somas))
        nome_soma = 'soma dos quadrados' if a.metodo == 'kmeans' else 'soma das distâncias'
        print(f'{nome_soma} de cada rodada, do menor para o maior valor:')
        for v in valores:
            quais = [str(i) for i, s in enumerate(somas, 1) if s == v]
            print(f'  {br(v, 1):>9}: {len(quais):>2} rodada(s) (sorteio {", ".join(quais)})')
        print(f'diferença entre a pior e a melhor: {br(100 * (max(somas) / min(somas) - 1), 1)}%')
        return

    if a.metodo == 'kmeans':
        grupos, centros, soma, rodadas = melhor_kmeans(z, a.k, a.inicios, a.sorteio)
        originais = np.array([x[grupos == j].mean(axis=0) for j in range(a.k)])
        print(f'K-means, K = {a.k}, {a.inicios} início(s) k-means++ (sorteio {a.sorteio}); o melhor parou na rodada {rodadas}')
        ordem = mostrar_grupos(ids, x, topo, grupos, originais)
        print()
        print(f'soma dos quadrados: {br(soma, 1)}')
    else:
        grupos, medoides, soma, rodadas = melhor_kmedoids(d, a.k, a.inicios, a.sorteio)
        originais = x[medoides]
        print(f'K-medoids (distância euclidiana), K = {a.k}, {a.inicios} início(s) k-medoids++ (sorteio {a.sorteio}); '
              f'o melhor parou na rodada {rodadas}')
        ordem = mostrar_grupos(ids, x, topo, grupos, originais, medoides)
        print()
        print(f'soma das distâncias até o medoide: {br(soma, 1)}')
    print('grupos numerados pelo número de pedidos do centro, do maior para o menor')

    if a.salvar:
        numero = {j: n for n, j in enumerate(ordem, 1)}
        pd.DataFrame({'id_cliente': ids, 'grupo': [numero[g] for g in grupos]}).to_csv(a.salvar, index=False)
        print(f'gravei {a.salvar} ({len(ids)} linhas: id_cliente, grupo)')


if __name__ == '__main__':
    try:
        main(sys.argv[1:])
    except Erro as e:
        print(f'erro: {e}', file=sys.stderr)
        sys.exit(2)
