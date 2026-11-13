"""PCA nas colunas dos clientes da Loja Exemplo, só com o NumPy e o pandas.

    python3 pca_clientes.py                       # oito colunas, escala z
    python3 pca_clientes.py --sem-escala          # só centraliza, sem escala
    python3 pca_clientes.py --componentes 2       # reconstrói com 2 componentes
    python3 pca_clientes.py --metades             # muda a amostra 20 vezes e mede os ângulos
    python3 pca_clientes.py --log pedidos_12m,gasto_12m,ticket_medio
    python3 pca_clientes.py --colunas pedidos_12m,gasto_12m,dias_desde_ultima_compra
    python3 pca_clientes.py --sem-maiores-gastos 30

As opções se combinam. Rode de dentro da pasta da aula.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ARQUIVO = Path('..') / 'dados' / 'clientes' / 'clientes.csv'
PADRAO = ['pedidos_12m', 'gasto_12m', 'ticket_medio', 'dias_desde_ultima_compra',
          'categorias_distintas', 'pct_pedidos_cupom', 'pct_pedidos_app', 'devolucoes_12m']
TEXTO = {'data_cadastro', 'regiao'}
SORTEIOS = 20
SORTEIO_INICIAL = 2026
PESOS_MOSTRADOS = 4


class Erro(Exception):
    pass


def br(x, casas=2):
    x = round(float(x), casas) + 0.0  # sem o -0,00
    texto = f'{x:,.{casas}f}' if abs(x) >= 10000 else f'{x:.{casas}f}'
    return texto.replace(',', '_').replace('.', ',').replace('_', '.')


def reais(x):
    inteiro, centavos = f'{x:,.2f}'.split('.')
    return 'R$ ' + inteiro.replace(',', '.') + ',' + centavos


def lista(texto):
    return [c.strip() for c in texto.split(',') if c.strip()]


def ler_argumentos(argv):
    p = argparse.ArgumentParser(description='PCA nas colunas dos clientes da Loja Exemplo.')
    p.add_argument('--colunas', type=lista, default=PADRAO,
                   help='colunas separadas por vírgula (padrão: as oito de comportamento)')
    p.add_argument('--sem-escala', action='store_true', help='só centraliza; não divide pelo desvio')
    p.add_argument('--log', type=lista, default=[], help='colunas que passam pelo log10 antes da escala')
    p.add_argument('--componentes', type=int, help='quantos componentes guardar na reconstrução')
    p.add_argument('--metades', action='store_true', help='mede a estabilidade em 20 sorteios de duas metades')
    p.add_argument('--sem-maiores-gastos', type=int, default=0, metavar='N',
                   help='tira os N clientes de maior gasto_12m antes de tudo')
    p.add_argument('--sem-vazios', action='store_true',
                   help='tira os clientes com valor vazio nas colunas escolhidas')
    return p.parse_args(argv)


def preparar(clientes, a):
    colunas = list(dict.fromkeys(a.colunas))
    if len(colunas) < 2:
        raise Erro('escolha pelo menos duas colunas diferentes')
    desconhecidas = [c for c in colunas if c not in clientes.columns]
    if desconhecidas:
        raise Erro(f'coluna que não existe: {", ".join(desconhecidas)}. Colunas: {", ".join(clientes.columns)}')
    texto = [c for c in colunas if c in TEXTO]
    if texto:
        raise Erro(f'{", ".join(texto)} não é número. O PCA só usa colunas numéricas.')
    if a.componentes is not None and not 1 <= a.componentes <= len(colunas):
        raise Erro(f'--componentes precisa ficar entre 1 e {len(colunas)}, o número de colunas')
    fora = [c for c in a.log if c not in colunas]
    if fora:
        raise Erro(f'--log com coluna que não está em --colunas: {", ".join(fora)}')
    if a.sem_maiores_gastos < 0 or a.sem_maiores_gastos >= len(clientes) - 2:
        raise Erro('--sem-maiores-gastos precisa ser um número entre 0 e o total de clientes menos 3')
    if a.sem_maiores_gastos:
        clientes = clientes.drop(clientes.nlargest(a.sem_maiores_gastos, 'gasto_12m').index)
    vazios = clientes[colunas].isna().sum()
    vazios = vazios[vazios > 0]
    if len(vazios):
        detalhe = ', '.join(f'{c} tem {n} vazios' for c, n in vazios.items())
        if not a.sem_vazios:
            raise Erro(f'{detalhe}. O PCA não aceita valor vazio: tire a coluna ou use --sem-vazios '
                       'para tirar esses clientes (e diga quantos saíram).')
        clientes = clientes.dropna(subset=colunas)
    for c in a.log:
        if (clientes[c] <= 0).any():
            raise Erro(f'{c} tem valor zero ou negativo, e o log10 não existe para eles. Use --log só em '
                       'colunas sempre positivas.')
    return clientes.reset_index(drop=True), colunas


def matriz(clientes, colunas, log, sem_escala):
    x = clientes[colunas].astype(float).copy()
    for c in log:
        x[c] = np.log10(x[c])
    x = x.to_numpy()
    media = x.mean(axis=0)
    desvio = np.ones(x.shape[1]) if sem_escala else x.std(axis=0, ddof=1)
    if (desvio == 0).any():
        raise Erro('uma coluna tem o mesmo valor em todos os clientes; tire essa coluna')
    return (x - media) / desvio


def pca(z):
    """Retorna a variância de cada componente e as direções (uma por linha), maior primeiro."""
    _, s, vt = np.linalg.svd(z, full_matrices=False)
    variancias = s ** 2 / (len(z) - 1)
    for i in range(len(vt)):
        if vt[i][np.argmax(np.abs(vt[i]))] < 0:
            vt[i] = -vt[i]
    return variancias, vt


def angulo(a, b):
    """Maior ângulo, em graus, entre o espaço das linhas de a e o das linhas de b."""
    cossenos = np.linalg.svd(np.dot(a, b.T), compute_uv=False)
    return float(np.degrees(np.arccos(np.clip(cossenos.min(), 0.0, 1.0))))


def mostrar_componentes(variancias, vt, colunas):
    total = variancias.sum()
    print(f'variância das colunas somada: {br(total, 2)}')
    print('componente  variância  % do total  % acumulada')
    acumulada = 0.0
    for i, v in enumerate(variancias):
        acumulada += v
        print(f'  {i + 1:>2}  {br(v, 4):>14}  {br(100 * v / total, 1):>8}%  {br(100 * acumulada / total, 1):>9}%')
    k = min(PESOS_MOSTRADOS, len(colunas))
    largura = max(len(c) for c in colunas) + 2
    print(f'peso de cada coluna nos componentes 1 a {k}')
    print(' ' * (largura + 2) + '  '.join(f'comp {i + 1}' for i in range(k)))
    for j, c in enumerate(colunas):
        print(f'  {c:<{largura}}' + '  '.join(f'{br(vt[i][j]):>6}' for i in range(k)))


def mostrar_reconstrucao(z, variancias, vt, colunas, clientes, k):
    p = len(colunas)
    guardadas = vt[:k]
    volta = np.dot(np.dot(z, guardadas.T), guardadas)
    erro = (z - volta) ** 2
    total = variancias.sum()
    fora = variancias[k:].sum()
    print()
    print(f'reconstrução com {k} componente{"s" if k > 1 else ""}')
    print(f'  fica de fora {br(100 * erro.sum() / (z ** 2).sum(), 1)}% da variância '
          f'(componentes {k + 1} a {p}: {br(fora, 4)} de {br(total, 4)})' if k < p else
          '  fica de fora 0,0% da variância: com todos os componentes, a tabela volta inteira')
    print('  quanto de cada coluna volta')
    largura = max(len(c) for c in colunas) + 2
    volta_coluna = 1 - erro.sum(axis=0) / (z ** 2).sum(axis=0)
    for c, f in zip(colunas, volta_coluna):
        print(f'    {c:<{largura}}{br(100 * f, 1):>6}%')
    por_cliente = erro.sum(axis=1)
    piores = np.argsort(-por_cliente, kind='stable')[:5]
    print(f'  5 clientes que voltam pior (erro do cliente; mediana dos {len(z)}: {br(float(np.median(por_cliente)))})')
    for i in piores:
        cl = clientes.iloc[i]
        dias = int(cl['dias_desde_ultima_compra'])
        print(f'    id {int(cl["id_cliente"])}: erro {br(por_cliente[i], 1)} | {int(cl["pedidos_12m"])} pedidos, '
              f'{reais(cl["gasto_12m"])}, ticket {reais(cl["ticket_medio"])}, '
              f'{dias} dia{"s" if dias != 1 else ""} sem comprar, devoluções: {int(cl["devolucoes_12m"])}')


def mostrar_metades(clientes, colunas, a):
    sorteio = np.random.default_rng(SORTEIO_INICIAL)
    p = len(colunas)
    sozinho = np.zeros((SORTEIOS, p))
    juntos = np.zeros((SORTEIOS, p - 1))
    meio = len(clientes) // 2
    for t in range(SORTEIOS):
        ordem = sorteio.permutation(len(clientes))
        um = clientes.iloc[ordem[:meio]]
        outro = clientes.iloc[ordem[meio:]]
        _, va = pca(matriz(um, colunas, a.log, a.sem_escala))
        _, vb = pca(matriz(outro, colunas, a.log, a.sem_escala))
        sozinho[t] = [angulo(va[i:i + 1], vb[i:i + 1]) for i in range(p)]
        juntos[t] = [angulo(va[:k], vb[:k]) for k in range(1, p)]
    print()
    print(f'estabilidade: {SORTEIOS} sorteios dos clientes em duas metades, PCA em cada metade')
    print('ângulo entre a direção de uma metade e a da outra (0 = mesma direção, 90 = nada a ver)')
    print('componente  sozinho: mediana  máximo  |  componentes 1 até ele juntos: máximo')
    for i in range(p):
        junto = br(juntos[:, i].max(), 1) if i < p - 1 else 'todos (sempre 0)'
        print(f'  {i + 1:>2}  {br(np.median(sozinho[:, i]), 1):>20}  {br(sozinho[:, i].max(), 1):>7}  |  {junto:>8}')


def main(argv):
    a = ler_argumentos(argv)
    if not ARQUIVO.exists():
        print(f'erro: não achei {ARQUIVO}. Rode de dentro da pasta da aula.', file=sys.stderr)
        return 2
    try:
        brutos = pd.read_csv(ARQUIVO)
        clientes, colunas = preparar(brutos, a)
        z = matriz(clientes, colunas, a.log, a.sem_escala)
        variancias, vt = pca(z)
        saiu = len(brutos) - len(clientes)
        print(f'clientes: {len(clientes)}' + (f' ({saiu} tirados de {len(brutos)})' if saiu else ''))
        print(f'colunas ({len(colunas)}): {", ".join(colunas)}')
        if 'id_cliente' in colunas:
            print('atenção: id_cliente é o número do cadastro, na ordem de chegada. Ele não diz nada sobre como o '
                  'cliente compra.')
        if a.log:
            print(f'log10 antes da escala em: {", ".join(a.log)}')
        print('escala: nenhuma, só centralizada (cada coluna na própria unidade)' if a.sem_escala else
              'escala: z (cada coluna com média 0 e desvio 1)')
        mostrar_componentes(variancias, vt, colunas)
        if a.componentes is not None:
            mostrar_reconstrucao(z, variancias, vt, colunas, clientes, a.componentes)
        if a.metades:
            mostrar_metades(clientes, colunas, a)
    except Erro as e:
        print(f'erro: {e}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
