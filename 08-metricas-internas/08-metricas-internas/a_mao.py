"""Silhouette, Davies-Bouldin e Calinski-Harabasz em poucos clientes, passo a passo.

    python3 a_mao.py                                                  # os 12 clientes e as distâncias
    python3 a_mao.py --grupos 27,78,104,1/15,23,105,137,77/63,98,143  # as três notas, com as contas
    python3 a_mao.py --grupos 27,78,104,1/15,23,105,137,77/63,98,143 --cliente 105
    python3 a_mao.py --grupos 27,78,104,1/15,23,105/63,98,143 --ruido 137,77
    python3 a_mao.py --grupos 27,78,104,1/15,23,105,137,77/63,98,143 --escala reais

Os grupos vão separados por barra, e os clientes de cada grupo por vírgula.
Duas colunas: pedidos no ano e ticket médio. Sem --escala, o ticket entra em centenas de reais,
com uma casa: R$ 424,06 vira 4,2.
Rode de dentro da pasta da aula.
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_samples, silhouette_score

ARQUIVO = Path('..') / 'dados' / 'clientes' / 'clientes.csv'
PADRAO = [27, 78, 104, 1, 15, 23, 105, 137, 63, 98, 143, 77]
LETRAS = 'ABCDEFGHIJ'
ESCALAS = {'centenas': 'ticket (R$ 100)', 'reais': 'ticket (R$)', 'z': 'ticket (z)'}


class Erro(Exception):
    pass


def br(x, casas=2):
    texto = f'{float(x):.{casas}f}'.replace('.', ',')
    return texto[1:] if texto.startswith('-') and set(texto[1:]) <= set('0,') else texto


def lista_ids(texto):
    try:
        ids = [int(p) for p in texto.split(',') if p.strip()]
    except ValueError:
        raise argparse.ArgumentTypeError(f'"{texto}" precisa ser uma lista de ids separados por vírgula')
    if not ids:
        raise argparse.ArgumentTypeError('a lista de ids está vazia')
    return ids


def lista_grupos(texto):
    partes = texto.split('/')
    if any(not p.strip() for p in partes):
        raise argparse.ArgumentTypeError('tem grupo vazio em --grupos: use barra entre grupos e vírgula entre clientes')
    return [lista_ids(p) for p in partes]


def ler_argumentos(argv):
    p = argparse.ArgumentParser(description='Silhouette, Davies-Bouldin e Calinski-Harabasz em poucos clientes.')
    p.add_argument('--grupos', type=lista_grupos, help='os grupos: clientes separados por vírgula, grupos por barra')
    p.add_argument('--ruido', type=lista_ids, help='clientes que ficaram fora de grupo (o ruído do DBSCAN)')
    p.add_argument('--cliente', type=int, help='com --grupos: mostra a conta da silhouette desse cliente, distância por distância')
    p.add_argument('--escala', choices=list(ESCALAS), default='centenas', help='como o ticket entra na conta')
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
    if (a.ruido or a.cliente is not None) and not a.grupos:
        raise Erro('--ruido e --cliente só valem com --grupos')
    if not a.grupos:
        return ids, None
    listados = [i for g in a.grupos for i in g] + list(a.ruido or [])
    repetidos = sorted({i for i in listados if listados.count(i) > 1})
    if repetidos:
        raise Erro(f'cliente em mais de um lugar em --grupos/--ruido: {repetidos}')
    estranhos = [i for i in listados if i not in ids]
    if estranhos:
        raise Erro(f'--grupos/--ruido têm clientes que não estão na conta: {estranhos}')
    sem_lugar = [i for i in ids if i not in listados]
    if sem_lugar:
        raise Erro(f'esses clientes não estão em nenhum grupo nem no --ruido: {sem_lugar}')
    if not 2 <= len(a.grupos) <= len(LETRAS):
        raise Erro(f'--grupos precisa de 2 a {len(LETRAS)} grupos; vieram {len(a.grupos)}')
    if a.cliente is not None and a.cliente not in [i for g in a.grupos for i in g]:
        raise Erro(f'o cliente {a.cliente} do --cliente não está em nenhum grupo')
    if len([i for g in a.grupos for i in g]) <= len(a.grupos):
        raise Erro('as notas precisam de mais clientes do que grupos')
    rotulos = np.full(len(ids), -1)
    for n, g in enumerate(a.grupos):
        for i in g:
            rotulos[ids.index(i)] = n
    return ids, rotulos


def ler_clientes():
    if not ARQUIVO.is_file():
        raise Erro(f'não achei {ARQUIVO}. Rode de dentro da pasta da aula, 08-metricas-internas.')
    return pd.read_csv(ARQUIVO).set_index('id_cliente')


def tabela(base, ids, escala):
    pedidos = base.loc[ids, 'pedidos_12m'].to_numpy(float)
    ticket = base.loc[ids, 'ticket_medio'].to_numpy(float)
    if escala == 'centenas':
        return np.c_[pedidos, np.round(ticket / 100, 1)]
    if escala == 'reais':
        return np.c_[pedidos, ticket]
    todos = base[['pedidos_12m', 'ticket_medio']].to_numpy(float)
    media, desvio = todos.mean(axis=0), todos.std(axis=0, ddof=1)
    return (np.c_[pedidos, ticket] - media) / desvio


def casas(escala):
    return {'centenas': 1, 'reais': 2, 'z': 2}[escala]


def titulo(texto):
    print(texto)
    print('-' * len(texto))


def mostrar_clientes(ids, x, escala):
    print(f'cliente      pedidos  {ESCALAS[escala]:>16}')
    for i, (p, t) in zip(ids, x):
        pedidos = f'{int(p)}' if escala != 'z' else br(p)
        print(f'{i:>7} {pedidos:>12} {br(t, casas(escala)):>16}')


def mostrar_distancias(ids, d):
    print()
    titulo('Distância entre cada par de clientes')
    largura = max(7, max(len(br(v, 1)) for v in d.ravel()) + 1)
    print('cliente' + ''.join(f'{i:>{largura}}' for i in ids))
    for i, linha in zip(ids, d):
        print(f'{i:>7}' + ''.join(f'{br(v, 1):>{largura}}' for v in linha))


def nomes(ids, membros):
    return ', '.join(str(ids[j]) for j in membros)


def silhouette_a_mao(d, rotulos):
    """Para cada cliente em grupo: a, b, o grupo vizinho e s. Grupo de um cliente só tem s = 0."""
    grupos = sorted(set(rotulos) - {-1})
    linhas = []
    for j in range(len(rotulos)):
        g = rotulos[j]
        if g == -1:
            linhas.append(None)
            continue
        mesmo = [m for m in range(len(rotulos)) if rotulos[m] == g and m != j]
        if not mesmo:
            linhas.append((None, None, None, 0.0))
            continue
        a = d[j, mesmo].mean()
        medias = {h: d[j, [m for m in range(len(rotulos)) if rotulos[m] == h]].mean() for h in grupos if h != g}
        vizinho = min(medias, key=medias.get)
        b = medias[vizinho]
        linhas.append((a, b, vizinho, (b - a) / max(a, b)))
    return linhas


def davies_bouldin_a_mao(x, rotulos):
    grupos = sorted(set(rotulos))
    centros = np.array([x[rotulos == g].mean(axis=0) for g in grupos])
    espalhamento = np.array([np.linalg.norm(x[rotulos == g] - c, axis=1).mean() for g, c in zip(grupos, centros)])
    m = cdist(centros, centros)
    r = np.full((len(grupos), len(grupos)), np.nan)
    for i in range(len(grupos)):
        for j in range(len(grupos)):
            if i != j:
                r[i, j] = (espalhamento[i] + espalhamento[j]) / m[i, j]
    pior = np.nanmax(r, axis=1)
    return centros, espalhamento, m, r, pior, pior.mean()


def calinski_a_mao(x, rotulos):
    grupos = sorted(set(rotulos))
    geral = x.mean(axis=0)
    entre = sum((rotulos == g).sum() * ((x[rotulos == g].mean(axis=0) - geral) ** 2).sum() for g in grupos)
    dentro = sum(((x[rotulos == g] - x[rotulos == g].mean(axis=0)) ** 2).sum() for g in grupos)
    n, k = len(x), len(grupos)
    return geral, entre, dentro, n, k, (entre / (k - 1)) / (dentro / (n - k))


def mostrar_silhouette(ids, d, rotulos):
    titulo('Silhouette: a de cada cliente')
    print('a = distância média até os outros do mesmo grupo; b = distância média até o grupo vizinho, o mais perto')
    print('s = (b - a) / o maior dos dois. Cliente sozinho no grupo fica com s = 0.')
    print('cliente  grupo        a        b  vizinho        s')
    linhas = silhouette_a_mao(d, rotulos)
    for i, g, linha in zip(ids, rotulos, linhas):
        if linha is None:
            continue
        a, b, viz, s = linha
        if a is None:
            print(f'{i:>7}  {LETRAS[g]:>5}  {"-":>7}  {"-":>7}  {"-":>7}  {br(s):>7}   (sozinho no grupo)')
        else:
            print(f'{i:>7}  {LETRAS[g]:>5}  {br(a):>7}  {br(b):>7}  {LETRAS[viz]:>7}  {br(s):>7}')
    validos = [l[3] for l in linhas if l is not None]
    print('média de cada grupo: ' + '; '.join(
        f'{LETRAS[g]} {br(np.mean([l[3] for l, r in zip(linhas, rotulos) if r == g]))}' for g in sorted(set(rotulos) - {-1})))
    print(f'silhouette (a média dos {len(validos)} clientes): {br(np.mean(validos), 3)}')
    return np.mean(validos)


def mostrar_davies(x, rotulos):
    centros, esp, m, r, pior, db = davies_bouldin_a_mao(x, rotulos)
    grupos = sorted(set(rotulos))
    print()
    titulo('Davies-Bouldin: quanto cada grupo se confunde com o vizinho mais parecido')
    print('centro de cada grupo (a média) e espalhamento (a distância média dos clientes até o centro):')
    for g, c, e in zip(grupos, centros, esp):
        print(f'  {LETRAS[g]}: centro ({br(c[0])}; {br(c[1])}), espalhamento {br(e)}')
    print('distância entre os centros:')
    print('      ' + ''.join(f'{LETRAS[g]:>9}' for g in grupos))
    for g, linha in zip(grupos, m):
        print(f'  {LETRAS[g]:>3} ' + ''.join(f'{br(v):>9}' for v in linha))
    print('para cada par: (espalhamento de um + espalhamento do outro) / distância entre os centros')
    for i, g in enumerate(grupos):
        partes = [f'com {LETRAS[h]} {br(r[i, j], 3)}' for j, h in enumerate(grupos) if j != i]
        print(f'  {LETRAS[g]}: ' + '; '.join(partes) + f'. O pior: {br(pior[i], 3)}')
    print(f'Davies-Bouldin (a média dos piores): {br(db, 3)}')
    return db


def mostrar_calinski(x, rotulos):
    geral, entre, dentro, n, k, ch = calinski_a_mao(x, rotulos)
    print()
    titulo('Calinski-Harabasz: o espalhamento entre os grupos contra o de dentro deles')
    print(f'centro geral: ({br(geral[0])}; {br(geral[1])})')
    print(f'entre os grupos: soma de (clientes do grupo x distância ao quadrado do centro do grupo ao centro geral) = {br(entre, 1)}')
    print(f'dentro dos grupos: soma das distâncias ao quadrado de cada cliente até o centro do grupo dele = {br(dentro, 1)}')
    print(f'({br(entre, 1)} / ({k} - 1)) / ({br(dentro, 1)} / ({n} - {k})) = {br(ch, 1)}')
    return ch


def conferir(x, rotulos, s, db, ch):
    ok = (np.isclose(s, silhouette_score(x, rotulos)) and np.isclose(db, davies_bouldin_score(x, rotulos))
          and np.isclose(ch, calinski_harabasz_score(x, rotulos)))
    print()
    print('as funções silhouette_score, davies_bouldin_score e calinski_harabasz_score do scikit-learn '
          f'deram os mesmos números: {"sim" if ok else "NÃO"}')
    return ok


def mostrar_notas(ids, x, rotulos, escala):
    d = squareform(pdist(x))
    grupos = sorted(set(rotulos) - {-1})
    titulo('Grupos')
    for g in grupos:
        print(f'  {LETRAS[g]}: {nomes(ids, np.where(rotulos == g)[0])}')
    ruido = np.where(rotulos == -1)[0]
    if len(ruido):
        print(f'  ruído: {nomes(ids, ruido)}')
    print()
    dentro = rotulos >= 0
    xd, rd = x[dentro], rotulos[dentro]
    s = mostrar_silhouette([i for i, r in zip(ids, rotulos) if r >= 0], squareform(pdist(xd)), rd)
    db = mostrar_davies(xd, rd)
    ch = mostrar_calinski(xd, rd)
    ok = conferir(xd, rd, s, db, ch)
    if len(ruido):
        print()
        titulo(f'E o ruído? As notas acima deixaram de fora os {len(ruido)} clientes do ruído')
        print('Se o ruído entrar como se fosse mais um grupo (o rótulo -1 do DBSCAN passado direto às funções):')
        print(f'  silhouette {br(silhouette_score(x, rotulos), 3)}; Davies-Bouldin {br(davies_bouldin_score(x, rotulos), 3)}; '
              f'Calinski-Harabasz {br(calinski_harabasz_score(x, rotulos), 1)}')
    return ok, d


def mostrar_cliente(ids, d, rotulos, cliente):
    j = ids.index(cliente)
    g = rotulos[j]
    print()
    titulo(f'A silhouette do cliente {cliente}, distância por distância')
    for h in sorted(set(rotulos) - {-1}):
        membros = [m for m in range(len(ids)) if rotulos[m] == h and m != j]
        if not membros:
            print(f'  grupo {LETRAS[h]}: nenhum outro cliente')
            continue
        contas = ' + '.join(br(d[j, m], 1) for m in membros)
        nota = ' (o grupo dele)' if h == g else ''
        print(f'  grupo {LETRAS[h]}{nota}: ({contas}) / {len(membros)} = {br(d[j, membros].mean())}')
    a, b, viz, s = silhouette_a_mao(d, rotulos)[j]
    if a is None:
        print(f'  ele está sozinho no grupo: s = 0')
        return
    print(f'  a = {br(a)}; b = {br(b)} (grupo {LETRAS[viz]}); s = ({br(b)} - {br(a)}) / {br(max(a, b))} = {br(s)}')


def main(argv):
    a = ler_argumentos(argv)
    try:
        base = ler_clientes()
        ids, rotulos = validar(a, base)
        x = tabela(base, ids, a.escala)
        if rotulos is None:
            mostrar_clientes(ids, x, a.escala)
            mostrar_distancias(ids, squareform(pdist(x)))
            return 0
        ok, d = mostrar_notas(ids, x, rotulos, a.escala)
        if a.cliente is not None:
            dentro = rotulos >= 0
            ids_d = [i for i, r in zip(ids, rotulos) if r >= 0]
            mostrar_cliente(ids_d, squareform(pdist(x[dentro])), rotulos[dentro], a.cliente)
        return 0 if ok else 1
    except Erro as e:
        print(f'erro: {e}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    warnings.simplefilter('error')
    sys.exit(main(sys.argv[1:]))
