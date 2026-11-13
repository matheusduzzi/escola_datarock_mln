"""Segmentar os 3000 clientes da Loja Exemplo: escolher as colunas, a escala e o método, e ler o perfil dos grupos.

    python3 segmentar.py --colunas pedidos,gasto,ticket,dias,categorias,cupom,app,devolucoes --escala z
    python3 segmentar.py --colunas pedidos,ticket,dias,categorias,cupom,app --escala log --topo separar --perfil
    python3 segmentar.py --colunas pedidos,ticket,dias,categorias,cupom,app --escala log --topo separar --grupo 3
    python3 segmentar.py --colunas pedidos,ticket,dias,categorias,cupom,app --escala log --topo separar --metodo ward
    python3 segmentar.py --colunas pedidos,ticket,dias,categorias,cupom,app --escala log --topo separar --salvar meus.csv
    python3 segmentar.py --colunas pedidos,ticket,dias,categorias,cupom,app --escala z --topo separar --comparar meus.csv

As colunas têm nomes curtos: pedidos, gasto, ticket, dias, categorias, cupom, app, devolucoes, avaliacao e id.
Rode de dentro da pasta da aula.
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import HDBSCAN, AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score

PASTA_DADOS = Path('..') / 'dados' / 'clientes'
ARQUIVO = PASTA_DADOS / 'clientes.csv'
NOMES = {'pedidos': 'pedidos_12m', 'gasto': 'gasto_12m', 'ticket': 'ticket_medio',
         'dias': 'dias_desde_ultima_compra', 'categorias': 'categorias_distintas', 'cupom': 'pct_pedidos_cupom',
         'app': 'pct_pedidos_app', 'devolucoes': 'devolucoes_12m', 'avaliacao': 'avaliacao_media', 'id': 'id_cliente'}
TEXTO = {'regiao': 'a região é texto, não tem distância entre Sudeste e Norte; ela aparece no --perfil',
         'cadastro': 'a data de cadastro é uma data; os anos de casa aparecem no --perfil'}
COM_LOG = ['pedidos', 'gasto', 'ticket', 'devolucoes']
TOPO = 40
DATA_CORTE = pd.Timestamp('2026-06-30')
METODOS = {'kmeans': 'K-means', 'ward': 'Ward', 'hdbscan': 'HDBSCAN'}


class Erro(Exception):
    pass


def br(x, casas=1):
    texto = f'{float(x):,.{casas}f}'.replace(',', '_').replace('.', ',').replace('_', '.')
    return texto[1:] if texto.startswith('-') and set(texto[1:]) <= set('0,.') else texto


def ler_argumentos(argv):
    p = argparse.ArgumentParser(description='Segmentar os clientes da Loja Exemplo.')
    p.add_argument('--colunas', required=True, help='as colunas que entram na distância, separadas por vírgula')
    p.add_argument('--escala', required=True, choices=['z', 'log', 'robusta', 'nenhuma'],
                   help='z; log (log nos pedidos, gasto, ticket e devoluções, depois z); robusta (mediana e '
                        'intervalo entre quartis); nenhuma (como estão no arquivo)')
    p.add_argument('--topo', choices=['manter', 'separar'], default='manter',
                   help=f'separar: tira os clientes de {TOPO} pedidos ou mais antes de agrupar (padrão: manter)')
    p.add_argument('--vazio', choices=['media', 'zero'], help='com avaliacao: o que pôr no lugar da avaliação vazia')
    p.add_argument('--metodo', choices=list(METODOS), default='kmeans', help='kmeans, ward ou hdbscan (padrão kmeans)')
    p.add_argument('--k', type=int, help='quantos grupos no kmeans e no ward, de 2 a 10 (padrão 4)')
    p.add_argument('--min-grupo', type=int, help='hdbscan: tamanho mínimo de um grupo, de 5 a 500 (padrão 50)')
    p.add_argument('--ver-escala', action='store_true', help='como cada coluna fica na escala escolhida')
    p.add_argument('--perfil', action='store_true', help='o perfil de cada grupo, com tamanho e números')
    p.add_argument('--grupo', type=int, help='a faixa do meio de cada coluna num grupo só')
    p.add_argument('--gerador', action='store_true', help='cruza os grupos com os perfis que o gerador sorteou')
    p.add_argument('--salvar', metavar='ARQUIVO.csv', help='grava id_cliente e grupo de cada cliente')
    p.add_argument('--comparar', metavar='ARQUIVO.csv', help='compara com um resultado gravado pelo --salvar')
    return p.parse_args(argv)


def validar(a):
    nomes = [n.strip() for n in a.colunas.split(',') if n.strip()]
    for n in nomes:
        if n in TEXTO:
            raise Erro(f'{n} não entra na distância: {TEXTO[n]}')
        if n not in NOMES:
            raise Erro(f'não conheço a coluna "{n}". Use: {", ".join(NOMES)}')
    if len(set(nomes)) != len(nomes):
        raise Erro('coluna repetida em --colunas')
    if len(nomes) < 2:
        raise Erro('use pelo menos 2 colunas em --colunas')
    if 'avaliacao' in nomes and not a.vazio:
        raise Erro('a avaliação fica vazia para quem nunca avaliou nada. Com avaliacao, diga o que pôr no vazio: '
                   '--vazio media ou --vazio zero')
    if a.vazio and 'avaliacao' not in nomes:
        raise Erro('--vazio só vale com a coluna avaliacao')
    if a.k is not None and a.metodo == 'hdbscan':
        raise Erro('o hdbscan não recebe --k: ele decide quantos grupos. Use --min-grupo')
    if a.k is not None and not 2 <= a.k <= 10:
        raise Erro('--k precisa ficar entre 2 e 10')
    if a.min_grupo is not None and a.metodo != 'hdbscan':
        raise Erro('--min-grupo só vale com --metodo hdbscan')
    if a.min_grupo is not None and not 5 <= a.min_grupo <= 500:
        raise Erro('--min-grupo precisa ficar entre 5 e 500')
    for opcao, valor in (('--salvar', a.salvar), ('--comparar', a.comparar)):
        if valor and not valor.endswith('.csv'):
            raise Erro(f'{opcao} precisa de um nome que termine em .csv')
    if a.comparar and not Path(a.comparar).is_file():
        raise Erro(f'não achei {a.comparar}. Grave antes com --salvar')
    if a.comparar:
        try:
            colunas = list(pd.read_csv(a.comparar, nrows=0).columns)
        except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError):
            colunas = []
        if colunas != ['id_cliente', 'grupo']:
            raise Erro(f'{a.comparar} não veio do --salvar: as colunas precisam ser id_cliente e grupo')
    return nomes


def ler_clientes():
    if not ARQUIVO.is_file():
        raise Erro(f'não achei {ARQUIVO}. Rode de dentro da pasta da aula, 09-segmentar-clientes.')
    return pd.read_csv(ARQUIVO)


def representar(base, nomes, escala, vazio):
    """A tabela que o método recebe: uma linha por cliente, uma coluna por nome, na escala pedida."""
    x = pd.DataFrame({n: base[NOMES[n]].astype(float) for n in nomes})
    if 'avaliacao' in nomes:
        x['avaliacao'] = x['avaliacao'].fillna(x['avaliacao'].mean() if vazio == 'media' else 0.0)
    if escala == 'log':
        for n in nomes:
            if n in COM_LOG:
                x[n] = np.log1p(x[n])
    m = x.to_numpy(float)
    if escala in ('z', 'log'):
        desvio = m.std(axis=0, ddof=1)
        if (desvio == 0).any():
            raise Erro('uma das colunas tem o mesmo valor para todos os clientes')
        return (m - m.mean(axis=0)) / desvio
    if escala == 'robusta':
        q1, mediana, q3 = np.percentile(m, [25, 50, 75], axis=0)
        iqr = q3 - q1
        zerados = [n for n, v in zip(nomes, iqr) if v == 0]
        if zerados:
            raise Erro(f'na escala robusta, {", ".join(zerados)} tem intervalo entre quartis 0: o primeiro e o '
                       'terceiro quartil são iguais, e a divisão não existe. Tire a coluna ou use outra escala')
        return (m - mediana) / iqr
    return m


def agrupar(x, metodo, k, minimo):
    if metodo == 'kmeans':
        return KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(x)
    if metodo == 'ward':
        return AgglomerativeClustering(n_clusters=k, linkage='ward').fit_predict(x)
    return HDBSCAN(min_cluster_size=minimo, copy=True).fit_predict(x)


def em_ordem(rotulos):
    """Renumera os grupos do maior para o menor, a partir de 1; quem ficou sem grupo continua -1."""
    tamanhos = pd.Series(rotulos[rotulos >= 0]).value_counts()
    ordem = sorted(tamanhos.index, key=lambda g: (-tamanhos[g], g))
    mapa = {g: n + 1 for n, g in enumerate(ordem)}
    return np.array([mapa.get(r, -1) for r in rotulos])


def titulo(texto):
    print(texto)
    print('-' * len(texto))


def descrever(nomes, a, k, minimo, n_topo, n_agrupados, n_vazio):
    titulo('Segmentação dos clientes da Loja Exemplo')
    print(f'colunas na distância: {", ".join(nomes)}')
    if a.escala == 'log':
        com = [n for n in nomes if n in COM_LOG]
        print(f'escala: log em {", ".join(com) if com else "nenhuma das colunas escolhidas"}, depois z')
    else:
        print(f'escala: {a.escala}')
    if 'avaliacao' in nomes:
        print(f'avaliação vazia: {n_vazio} clientes, com {"a média de quem avaliou" if a.vazio == "media" else "zero"} no lugar')
    if a.topo == 'separar':
        print(f'clientes do topo ({TOPO} pedidos ou mais): {n_topo} separados antes de agrupar')
    else:
        print(f'clientes do topo ({TOPO} pedidos ou mais): {n_topo}, agrupados com os outros')
    if a.metodo == 'hdbscan':
        print(f'método: HDBSCAN, grupo mínimo de {minimo} clientes')
    elif a.metodo == 'kmeans':
        print(f'método: K-means, K = {k}, 10 começos')
    else:
        print(f'método: Ward, K = {k}')
    print(f'clientes agrupados: {n_agrupados}')


def mostrar_escala(x, nomes, escala):
    print()
    titulo(f'Como cada coluna ficou na escala {escala}, do menor ao maior cliente')
    print(f'{"coluna":<12}{"menor":>9}{"5%":>9}{"25%":>9}{"mediana":>9}{"75%":>9}{"95%":>9}{"maior":>9}')
    for j, n in enumerate(nomes):
        v = np.percentile(x[:, j], [0, 5, 25, 50, 75, 95, 100])
        print(f'{n:<12}' + ''.join(f'{br(q, 2):>9}' for q in v))


def mostrar_tamanhos(x, rotulos):
    print()
    print('grupo  clientes  % dos agrupados')
    n = len(rotulos)
    for g in sorted(set(rotulos[rotulos > 0])):
        q = int((rotulos == g).sum())
        print(f'{g:>5} {q:>9} {br(100 * q / n):>16}')
    sem = int((rotulos == -1).sum())
    if sem:
        print(f'{"sem grupo":>9}{sem:>5} {br(100 * sem / n):>16}')
    grupos = rotulos[rotulos > 0]
    if len(grupos):
        print(f'menor grupo: {int(pd.Series(grupos).value_counts().min())} clientes')
    dentro = rotulos > 0
    if len(set(rotulos[dentro])) >= 2:
        s = silhouette_score(x[dentro], rotulos[dentro])
        if sem:
            print(f'silhouette nessa representação, sem os {sem} sem grupo: {br(s, 3)}')
        else:
            print(f'silhouette nessa representação: {br(s, 3)}')
    elif len(set(rotulos[dentro])) == 1:
        print('silhouette: sem nota, sobrou um grupo só')
    else:
        print('silhouette: sem nota, nenhum cliente ficou em grupo')


def medidas(parte, total):
    """Os números do perfil de um conjunto de clientes."""
    dias = parte['dias_desde_ultima_compra']
    casa = (DATA_CORTE - pd.to_datetime(parte['data_cadastro'])).dt.days / 365.25
    regiao = parte['regiao'].value_counts()
    return [
        ('clientes', f'{len(parte)}'),
        ('% dos agrupados', br(100 * len(parte) / total)),
        ('pedidos', br(parte['pedidos_12m'].median())),
        ('gasto R$', br(parte['gasto_12m'].median(), 0)),
        ('ticket R$', br(parte['ticket_medio'].median(), 0)),
        ('dias sem comprar', br(dias.median(), 0)),
        ('categorias', br(parte['categorias_distintas'].median())),
        ('cupom %', br(parte['pct_pedidos_cupom'].median(), 0)),
        ('app %', br(parte['pct_pedidos_app'].median(), 0)),
        ('com devolução %', br(100 * (parte['devolucoes_12m'] > 0).mean(), 0)),
        ('nota de quem avaliou', br(parte['avaliacao_media'].median())),
        ('sem avaliação %', br(100 * parte['avaliacao_media'].isna().mean(), 0)),
        ('um pedido só %', br(100 * (parte['pedidos_12m'] == 1).mean(), 0)),
        ('mais de 180 dias', f'{int((dias > 180).sum())}'),
        ('anos de casa', br(casa.median())),
        ('região mais comum', f'{regiao.index[0]} {br(100 * regiao.iloc[0] / len(parte), 0)}%'),
    ]


def mostrar_perfil(base, x, rotulos, nomes, escala):
    grupos = sorted(set(rotulos[rotulos > 0]))
    partes = [(f'grupo {g}', base[rotulos == g]) for g in grupos]
    if (rotulos == -1).any():
        partes.append(('sem grupo', base[rotulos == -1]))
    partes.append(('todos', base))
    total = len(base)
    colunas = [(nome, medidas(parte, total)) for nome, parte in partes]
    largura = max(len(nome) for nome, _ in colunas[0][1]) + 2
    print()
    titulo('Perfil dos grupos (mediana, salvo onde diz %)')
    print(' ' * largura + ''.join(f'{nome:>15}' for nome, _ in colunas))
    for i, (rotulo, _) in enumerate(colunas[0][1]):
        print(f'{rotulo:<{largura}}' + ''.join(f'{m[i][1]:>15}' for _, m in colunas))
    print()
    titulo(f'O centro de cada grupo nas colunas da distância (escala {escala})')
    print(' ' * largura + ''.join(f'{f"grupo {g}":>15}' for g in grupos))
    centros = {g: x[rotulos == g].mean(axis=0) for g in grupos}
    for j, n in enumerate(nomes):
        print(f'{n:<{largura}}' + ''.join(f'{br(centros[g][j], 2):>15}' for g in grupos))
    if escala in ('z', 'log'):
        print()
        print('O que mais afasta cada grupo do centro de todos (as duas colunas mais longe do 0):')
        for g in grupos:
            ordem = np.argsort(-np.abs(centros[g]))[:2]
            print(f'  grupo {g}: ' + '; '.join(f'{nomes[j]} {br(centros[g][j], 2)}' for j in ordem))


def mostrar_grupo(base, rotulos, g):
    parte = base[rotulos == g]
    print()
    titulo(f'Grupo {g}: {len(parte)} clientes, a faixa do meio de cada coluna')
    print(f'{"coluna":<26}{"25%":>10}{"mediana":>10}{"75%":>10}{"maior":>10}{"todos: mediana":>17}')
    for n in ['pedidos', 'gasto', 'ticket', 'dias', 'categorias', 'cupom', 'app', 'devolucoes', 'avaliacao']:
        c = NOMES[n]
        q1, q2, q3 = parte[c].quantile([0.25, 0.5, 0.75])
        casas = 0 if n in ('gasto', 'ticket') else 1
        print(f'{c:<26}{br(q1, casas):>10}{br(q2, casas):>10}{br(q3, casas):>10}{br(parte[c].max(), casas):>10}'
              f'{br(base[c].median(), casas):>17}')
    print(f'clientes com 50% ou mais dos pedidos com cupom: {int((parte["pct_pedidos_cupom"] >= 50).sum())}')
    print(f'clientes com {TOPO} pedidos ou mais: {int((parte["pedidos_12m"] >= TOPO).sum())}')


def mostrar_gerador(base, rotulos):
    caminho = PASTA_DADOS / 'gerar.py'
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location('gerar_clientes', caminho)
    gerar = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gerar)
    perfis = pd.Series([p for _, p in gerar.sortear()], index=range(1, 3001))
    tabela = pd.crosstab(pd.Series(rotulos, name='grupo'), perfis.loc[base['id_cliente']].reset_index(drop=True).rename('perfil'))
    tabela.index = ['sem grupo' if g == -1 else f'grupo {g}' for g in tabela.index]
    print()
    titulo('Os grupos contra os perfis que o gerador sorteou (só depois de agrupar)')
    print(tabela.to_string())


def salvar(base, rotulos, topo, arquivo):
    saida = pd.DataFrame({'id_cliente': base['id_cliente'], 'grupo': [str(r) for r in rotulos]})
    if len(topo):
        saida = pd.concat([saida, pd.DataFrame({'id_cliente': topo['id_cliente'], 'grupo': 'topo'})])
    saida.sort_values('id_cliente').to_csv(arquivo, index=False)
    print(f'\ngravei {len(saida)} clientes em {arquivo}')


def comparar(base, rotulos, topo, arquivo, todos):
    antes = pd.read_csv(arquivo, dtype={'grupo': str})
    agora = pd.DataFrame({'id_cliente': base['id_cliente'], 'grupo': [str(r) for r in rotulos]})
    if len(topo):
        agora = pd.concat([agora, pd.DataFrame({'id_cliente': topo['id_cliente'], 'grupo': 'topo'})])
    junto = antes.merge(agora, on='id_cliente', suffixes=('_antes', '_agora'))
    tabela = pd.crosstab(junto['grupo_antes'], junto['grupo_agora'])
    linhas, colunas = linear_sum_assignment(-tabela.to_numpy())
    mesmos = int(tabela.to_numpy()[linhas, colunas].sum())
    print()
    titulo(f'Comparação com {arquivo} (linhas: o gravado; colunas: agora)')
    print(tabela.to_string())
    pares = ', '.join(f'{tabela.index[i]} com {tabela.columns[j]}' for i, j in zip(linhas, colunas))
    print(f'cada grupo gravado casado com o grupo de agora que mais divide clientes com ele: {pares}')
    casado = {tabela.index[i]: tabela.columns[j] for i, j in zip(linhas, colunas)}
    trocou = junto[[casado.get(a) != b for a, b in zip(junto['grupo_antes'], junto['grupo_agora'])]]
    print(f'clientes que trocam de grupo: {len(trocou)} de {len(junto)}')
    if len(trocou):
        quem = todos.set_index('id_cliente').loc[trocou['id_cliente']]
        print('quem trocou, mediana: ' + '; '.join(
            f'{rotulo} {br(quem[c].median(), casas)}' for rotulo, c, casas in
            [('pedidos', 'pedidos_12m', 1), ('ticket R$', 'ticket_medio', 0), ('dias sem comprar', 'dias_desde_ultima_compra', 0),
             ('categorias', 'categorias_distintas', 1), ('cupom %', 'pct_pedidos_cupom', 0), ('app %', 'pct_pedidos_app', 0)]))


def main(argv):
    a = ler_argumentos(argv)
    try:
        nomes = validar(a)
        base = ler_clientes()
        todos = base
        k = a.k or 4
        minimo = a.min_grupo or 50
        no_topo = base['pedidos_12m'] >= TOPO
        topo = base[no_topo] if a.topo == 'separar' else base.iloc[0:0]
        if a.topo == 'separar':
            base = base[~no_topo].reset_index(drop=True)
        x = representar(base, nomes, a.escala, a.vazio)
        rotulos = em_ordem(agrupar(x, a.metodo, k, minimo))
        if a.grupo is not None and a.grupo not in set(rotulos[rotulos > 0]):
            raise Erro(f'não existe o grupo {a.grupo}. Os grupos vão de 1 a {int(rotulos.max())}')
        descrever(nomes, a, k, minimo, int(no_topo.sum()), len(base), int(base['avaliacao_media'].isna().sum()))
        if a.ver_escala:
            mostrar_escala(x, nomes, a.escala)
        mostrar_tamanhos(x, rotulos)
        if a.perfil:
            mostrar_perfil(base, x, rotulos, nomes, a.escala)
        if a.grupo is not None:
            mostrar_grupo(base, rotulos, a.grupo)
        if a.gerador:
            mostrar_gerador(base, rotulos)
        if a.comparar:
            comparar(base, rotulos, topo, a.comparar, todos)
        if a.salvar:
            salvar(base, rotulos, topo, a.salvar)
    except Erro as e:
        print(f'erro: {e}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
