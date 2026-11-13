"""K-means e K-medoids em poucos clientes, rodada por rodada, para conferir a conta feita à mão.

    python3 a_mao.py                                   # os 12 clientes e as duas colunas
    python3 a_mao.py --inicio 1,105,143                # K-means a partir desses três clientes
    python3 a_mao.py --metodo kmedoids --inicio 1,105,143
    python3 a_mao.py --melhor 3                        # o melhor resultado com K = 3, entre todos os inícios
    python3 a_mao.py --curva                           # a soma de cada método para K de 1 a 6
    python3 a_mao.py --sem 77 --melhor 3               # os mesmos clientes, sem o cliente 77
    python3 a_mao.py --escala reais --melhor 3         # o ticket em reais, não em centenas de reais
    python3 a_mao.py --ids 2,7,15,23 --inicio 2,7      # outros clientes

As colunas são pedidos_12m e ticket_medio. O ticket entra em centenas de reais, com uma casa:
R$ 424,06 vira 4,2. Rode de dentro da pasta da aula.
"""
import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ARQUIVO = Path('..') / 'dados' / 'clientes' / 'clientes.csv'
CASO = [27, 78, 104, 1, 15, 23, 105, 137, 63, 98, 143, 77]
NOMES = {'kmeans': 'K-means', 'kmedoids': 'K-medoids'}
LETRAS = 'ABCDEF'
MAX_RODADAS = 50


class Erro(Exception):
    pass


def br(x, casas=1):
    texto = f'{float(x):.{casas}f}'.replace('.', ',')
    return texto[1:] if texto.startswith('-') and set(texto[1:]) <= set('0,') else texto


def lista_ids(texto):
    try:
        ids = [int(t) for t in texto.replace(' ', '').split(',') if t]
    except ValueError:
        raise argparse.ArgumentTypeError(f'"{texto}" precisa ser uma lista de ids separados por vírgula, como 1,105,143')
    if not ids:
        raise argparse.ArgumentTypeError('a lista de ids está vazia')
    return ids


def ler_argumentos(argv):
    p = argparse.ArgumentParser(description='K-means e K-medoids em poucos clientes, rodada por rodada.')
    p.add_argument('--metodo', choices=['kmeans', 'kmedoids'], default='kmeans')
    p.add_argument('--inicio', type=lista_ids, metavar='ID,ID', help='clientes que servem de centro no começo')
    p.add_argument('--melhor', type=int, metavar='K', help='testa todos os inícios e mostra o melhor resultado')
    p.add_argument('--curva', action='store_true', help='a soma de cada método para K de 1 a 6')
    p.add_argument('--ids', type=lista_ids, metavar='ID,ID', help='outros clientes no lugar dos 12 da aula')
    p.add_argument('--sem', type=lista_ids, metavar='ID,ID', help='tira esses clientes')
    p.add_argument('--escala', choices=['centenas', 'reais', 'z'], default='centenas',
                   help='como o ticket entra: centenas de reais (padrão), reais, ou as duas colunas na escala z')
    p.add_argument('--distancia', choices=['euclidiana', 'manhattan'], default='euclidiana',
                   help='distância do K-medoids (o K-means usa sempre a euclidiana)')
    return p.parse_args(argv)


def validar(a, base):
    ids = list(a.ids) if a.ids else list(CASO)
    if len(set(ids)) != len(ids):
        raise Erro('--ids tem cliente repetido')
    fora = [i for i in ids if i not in base.index]
    if fora:
        raise Erro(f'cliente(s) {fora} não existe(m) em clientes.csv (os ids vão de 1 a {base.index.max()})')
    if a.sem:
        faltam = [i for i in a.sem if i not in ids]
        if faltam:
            raise Erro(f'--sem: cliente(s) {faltam} não está(ão) entre os clientes usados')
        ids = [i for i in ids if i not in a.sem]
    if not 4 <= len(ids) <= 20:
        raise Erro(f'use de 4 a 20 clientes (agora são {len(ids)}); a conta é para fazer à mão')
    modos = sum([a.inicio is not None, a.melhor is not None, a.curva])
    if modos > 1:
        raise Erro('escolha só uma: --inicio, --melhor ou --curva')
    if a.distancia != 'euclidiana' and a.metodo != 'kmedoids' and not a.curva:
        raise Erro('--distancia manhattan só vale com --metodo kmedoids (o K-means usa a média, que é da euclidiana)')
    if a.inicio is not None:
        if len(set(a.inicio)) != len(a.inicio):
            raise Erro('--inicio tem cliente repetido')
        fora = [i for i in a.inicio if i not in ids]
        if fora:
            raise Erro(f'--inicio: cliente(s) {fora} não está(ão) entre os clientes usados')
        if not 1 <= len(a.inicio) <= len(LETRAS) or len(a.inicio) >= len(ids):
            raise Erro(f'--inicio precisa de 1 a {min(len(LETRAS), len(ids) - 1)} clientes, um por grupo')
    if a.melhor is not None and not 1 <= a.melhor <= min(len(LETRAS), len(ids) - 1):
        raise Erro(f'--melhor precisa de um K entre 1 e {min(len(LETRAS), len(ids) - 1)}')
    return ids


def ler_clientes():
    if not ARQUIVO.is_file():
        raise Erro(f'não achei {ARQUIVO}. Rode de dentro da pasta da aula, 06-k-means-e-k-medoids.')
    return pd.read_csv(ARQUIVO).set_index('id_cliente')


def tabela(base, ids, escala):
    pedidos = base['pedidos_12m'].to_numpy(float)
    ticket = base['ticket_medio'].to_numpy(float)
    if escala == 'centenas':
        x = np.c_[pedidos, np.round(ticket / 100, 1)]
    elif escala == 'reais':
        x = np.c_[pedidos, np.round(ticket, 0)]
    else:
        x = np.c_[pedidos, ticket]
        x = np.round((x - x.mean(axis=0)) / x.std(axis=0, ddof=1), 2)
    linhas = base.index.get_indexer(ids)
    return x[linhas]


def distancias(x, c, tipo='euclidiana'):
    dif = x[:, None, :] - c[None, :, :]
    if tipo == 'manhattan':
        return np.abs(dif).sum(axis=2)
    return np.sqrt((dif * dif).sum(axis=2))


def rotulo_ticket(escala):
    return {'centenas': 'ticket (R$ 100)', 'reais': 'ticket (R$)', 'z': 'ticket (z)'}[escala]


def rotulo_pedidos(escala):
    return 'pedidos (z)' if escala == 'z' else 'pedidos'


def casas_ticket(escala):
    return {'centenas': 1, 'reais': 0, 'z': 2}[escala]


# ---------- os dois métodos ----------

def kmeans(x, inicio):
    """Lloyd: cada cliente vai para o centro mais perto; cada centro vira a média do seu grupo."""
    centros = x[inicio].astype(float)
    rodadas, grupos = [], None
    for _ in range(MAX_RODADAS):
        d = distancias(x, centros)
        novos = d.argmin(axis=1)
        vazio = [j for j in range(len(centros)) if not (novos == j).any()]
        if vazio:
            raise Erro(f'o grupo {LETRAS[vazio[0]]} ficou vazio nesta rodada; comece de outros clientes')
        if grupos is not None and (novos == grupos).all():
            rodadas.append((centros, d, grupos, None))
            break
        grupos = novos
        centros_novos = np.array([x[grupos == j].mean(axis=0) for j in range(len(centros))])
        rodadas.append((centros, d, grupos, centros_novos))
        centros = centros_novos
    soma = float(((x - centros[grupos]) ** 2).sum())
    return rodadas, grupos, centros, soma


def kmedoids(x, inicio, tipo):
    """Alternado: cada cliente vai para o medoide mais perto; o medoide vira o cliente do grupo com a menor
    soma de distâncias até os outros do grupo."""
    d_todos = distancias(x, x, tipo)
    medoides = list(inicio)
    rodadas = []
    for _ in range(MAX_RODADAS):
        d = d_todos[:, medoides]
        grupos = d.argmin(axis=1)
        somas, novos = {}, []
        for j, m in enumerate(medoides):
            membros = np.flatnonzero(grupos == j)
            s = d_todos[np.ix_(membros, membros)].sum(axis=1)
            somas[j] = list(zip(membros, s))
            melhor = s.min()
            candidatos = [i for i, v in zip(membros, s) if np.isclose(v, melhor)]
            novos.append(m if m in candidatos else min(candidatos))
        rodadas.append((list(medoides), d, grupos, somas, list(novos)))
        if novos == medoides:
            break
        medoides = novos
    soma = float(d_todos[:, medoides].min(axis=1).sum())
    return rodadas, grupos, medoides, soma


def melhor_kmeans(x, k):
    """Roda o K-means a partir de cada combinação de k clientes e guarda o de menor soma."""
    resultados = {}
    for inicio in itertools.combinations(range(len(x)), k):
        try:
            _, grupos, centros, soma = kmeans(x, list(inicio))
        except Erro:
            continue
        chave = round(soma, 6)
        if chave not in resultados:
            resultados[chave] = [0, grupos, centros]
        resultados[chave][0] += 1
    return resultados


def melhor_kmedoids(x, k, tipo):
    """Testa todas as combinações de k medoides: com poucos clientes, dá para achar o melhor de verdade."""
    d_todos = distancias(x, x, tipo)
    melhor = None
    for comb in itertools.combinations(range(len(x)), k):
        soma = d_todos[:, comb].min(axis=1).sum()
        if melhor is None or soma < melhor[0] - 1e-9:
            melhor = (soma, list(comb))
    return float(melhor[0]), melhor[1], d_todos[:, melhor[1]].argmin(axis=1)


# ---------- saídas ----------

def titulo(texto):
    print(texto)
    print('-' * len(texto))


def mostrar_clientes(ids, x, escala):
    ct = casas_ticket(escala)
    print(f'cliente  {rotulo_pedidos(escala):>11}  {rotulo_ticket(escala):>15}')
    for i, linha in zip(ids, x):
        print(f'{i:>7}  {br(linha[0], 2 if escala == "z" else 0):>11}  {br(linha[1], ct):>15}')


def grupos_texto(ids, grupos, k):
    return '; '.join(f'{LETRAS[j]}: ' + ', '.join(str(ids[i]) for i in np.flatnonzero(grupos == j)) for j in range(k))


def mostrar_kmeans(ids, x, inicio, escala):
    rodadas, grupos, centros, soma = kmeans(x, inicio)
    ct = casas_ticket(escala)
    cp = 2
    print('K-means, começando com os centros em ' + ', '.join(f'{LETRAS[j]} = cliente {ids[i]}' for j, i in enumerate(inicio)))
    print()
    for r, (c, d, g, cn) in enumerate(rodadas, 1):
        titulo(f'Rodada {r}: cada cliente vai para o centro mais perto')
        print('centros: ' + '; '.join(f'{LETRAS[j]} ({br(c[j, 0], cp)}; {br(c[j, 1], cp)})' for j in range(len(c))))
        cab = '  '.join(f'dist. a {LETRAS[j]}' for j in range(len(c)))
        print(f'cliente  {rotulo_pedidos(escala):>11}  {rotulo_ticket(escala):>15}  {cab}  grupo')
        for i in range(len(x)):
            ds = '  '.join(f'{br(d[i, j], 1):>9}' for j in range(len(c)))
            print(f'{ids[i]:>7}  {br(x[i, 0], 2 if escala == "z" else 0):>11}  {br(x[i, 1], ct):>15}  {ds}  {LETRAS[g[i]]:>5}')
        if cn is None:
            break
        print('centro novo de cada grupo (a média dos clientes dele):')
        for j in range(len(c)):
            membros = [str(ids[i]) for i in np.flatnonzero(g == j)]
            print(f'  {LETRAS[j]}: pedidos {br(cn[j, 0], cp)}, ticket {br(cn[j, 1], cp)} '
                  f'({len(membros)} cliente{"s" if len(membros) > 1 else ""}: {", ".join(membros)})')
        print()
    print()
    print(f'Os grupos da rodada {len(rodadas)} são os mesmos da rodada {len(rodadas) - 1}: o K-means parou.')
    print('grupos: ' + grupos_texto(ids, grupos, len(inicio)))
    print(f'soma das distâncias ao quadrado até o centro do grupo: {br(soma, 1)}')


def mostrar_kmedoids(ids, x, inicio, escala, tipo):
    rodadas, grupos, medoides, soma = kmedoids(x, inicio, tipo)
    ct = casas_ticket(escala)
    print(f'K-medoids (distância {tipo}), começando com os medoides em '
          + ', '.join(f'{LETRAS[j]} = cliente {ids[i]}' for j, i in enumerate(inicio)))
    print()
    for r, (m, d, g, somas, novos) in enumerate(rodadas, 1):
        titulo(f'Rodada {r}: cada cliente vai para o medoide mais perto')
        print('medoides: ' + '; '.join(f'{LETRAS[j]} = cliente {ids[i]}' for j, i in enumerate(m)))
        cab = '  '.join(f'dist. a {LETRAS[j]}' for j in range(len(m)))
        print(f'cliente  {rotulo_pedidos(escala):>11}  {rotulo_ticket(escala):>15}  {cab}  grupo')
        for i in range(len(x)):
            ds = '  '.join(f'{br(d[i, j], 1):>9}' for j in range(len(m)))
            print(f'{ids[i]:>7}  {br(x[i, 0], 2 if escala == "z" else 0):>11}  {br(x[i, 1], ct):>15}  {ds}  {LETRAS[g[i]]:>5}')
        print('em cada grupo, a soma das distâncias de cada cliente até os outros do grupo:')
        for j in range(len(m)):
            partes = '; '.join(f'{ids[i]} soma {br(s, 1)}' for i, s in somas[j])
            print(f'  {LETRAS[j]}: {partes}. Medoide: cliente {ids[novos[j]]}')
        print()
    print(f'Os medoides que saem da rodada {len(rodadas)} são os mesmos que entraram nela: o K-medoids parou.')
    print('grupos: ' + grupos_texto(ids, grupos, len(inicio)))
    print('medoides: ' + ', '.join(f'{LETRAS[j]} = cliente {ids[i]}' for j, i in enumerate(medoides)))
    print(f'soma das distâncias até o medoide do grupo: {br(soma, 1)}')


def mostrar_melhor(ids, x, k, metodo, tipo):
    if metodo == 'kmeans':
        resultados = melhor_kmeans(x, k)
        total = sum(v[0] for v in resultados.values())
        print(f'K-means com K = {k}: rodei a partir de cada combinação de {k} clientes ({total} inícios).')
        print('resultados diferentes, do menor para o maior:')
        for soma in sorted(resultados):
            n, grupos, centros = resultados[soma]
            print(f'  soma {br(soma, 1)}: {n} de {total} inícios chegaram aqui')
            print(f'    grupos: {grupos_texto(ids, grupos, k)}')
            print('    centros: ' + '; '.join(f'{LETRAS[j]} ({br(centros[j, 0], 2)}; {br(centros[j, 1], 2)})' for j in range(k)))
    else:
        soma, medoides, grupos = melhor_kmedoids(x, k, tipo)
        n = len(list(itertools.combinations(range(len(x)), k)))
        print(f'K-medoids com K = {k} (distância {tipo}): testei as {n} combinações de {k} medoides.')
        print(f'  a menor soma das distâncias até o medoide: {br(soma, 1)}')
        print(f'    grupos: {grupos_texto(ids, grupos, k)}')
        print('    medoides: ' + ', '.join(f'{LETRAS[j]} = cliente {ids[i]}' for j, i in enumerate(medoides)))


def mostrar_curva(ids, x, tipo):
    print('Para cada K, o melhor resultado de cada método (todos os inícios no K-means, todas as combinações no K-medoids).')
    print(f'K  K-means: soma dos quadrados  queda   K-medoids: soma das distâncias ({tipo})  queda  grupos de 1 cliente')
    anterior = {}
    for k in range(1, min(6, len(x) - 1) + 1):
        km = melhor_kmeans(x, k)
        soma_km = min(km)
        _, grupos_km, _ = km[soma_km][0], km[soma_km][1], km[soma_km][2]
        soma_kd, _, grupos_kd = melhor_kmedoids(x, k, tipo)
        quedas = ['' if k == 1 else br(anterior[m] - s, 1) for m, s in (('km', soma_km), ('kd', soma_kd))]
        sozinhos_km = [str(ids[i]) for j in range(k) for i in np.flatnonzero(grupos_km == j) if (grupos_km == j).sum() == 1]
        sozinhos_kd = [str(ids[i]) for j in range(k) for i in np.flatnonzero(grupos_kd == j) if (grupos_kd == j).sum() == 1]
        so = f'K-means: {", ".join(sozinhos_km) or "nenhum"}; K-medoids: {", ".join(sozinhos_kd) or "nenhum"}'
        print(f'{k}  {br(soma_km, 1):>26}  {quedas[0]:>6}   {br(soma_kd, 1):>40}  {quedas[1]:>5}  {so}')
        anterior = {'km': soma_km, 'kd': soma_kd}


def main(argv):
    a = ler_argumentos(argv)
    base = ler_clientes()
    ids = validar(a, base)
    x = tabela(base, ids, a.escala)
    unidade = {'centenas': 'ticket em centenas de reais', 'reais': 'ticket em reais',
               'z': 'pedidos e ticket na escala z dos 3000 clientes'}[a.escala]
    print(f'clientes: {len(ids)}; colunas: pedidos_12m e ticket_medio ({unidade})')
    if a.sem:
        print('fora da conta: cliente ' + ', '.join(map(str, a.sem)))
    print()
    if a.curva:
        mostrar_curva(ids, x, a.distancia)
    elif a.melhor is not None:
        mostrar_melhor(ids, x, a.melhor, a.metodo, a.distancia)
    elif a.inicio is not None:
        inicio = [ids.index(i) for i in a.inicio]
        if a.metodo == 'kmeans':
            mostrar_kmeans(ids, x, inicio, a.escala)
        else:
            mostrar_kmedoids(ids, x, inicio, a.escala, a.distancia)
    else:
        mostrar_clientes(ids, x, a.escala)


if __name__ == '__main__':
    try:
        main(sys.argv[1:])
    except Erro as e:
        print(f'erro: {e}', file=sys.stderr)
        sys.exit(2)
