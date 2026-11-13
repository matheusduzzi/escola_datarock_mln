"""Ver se os grupos de agosto se mantêm e montar o teste da campanha.

A segmentação de partida é a da aula «Segmentar clientes»: seis colunas (pedidos, ticket, dias, categorias,
cupom, app), log nos pedidos e no ticket e depois a escala z, os clientes de 40 pedidos ou mais separados antes,
K-means com K = 4 e 10 começos. Cada opção abaixo muda uma dessas escolhas.

    python3 estabilidade.py partida
    python3 estabilidade.py partida --comecos 1
    python3 estabilidade.py amostra
    python3 estabilidade.py amostra --ver 3
    python3 estabilidade.py amostra --comecos 50 --fronteira
    python3 estabilidade.py comparar --k 3
    python3 estabilidade.py comparar --escala z
    python3 estabilidade.py teste
    python3 estabilidade.py teste --salvar envio-agosto.csv

Rode de dentro da pasta da aula.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

ARQUIVO = Path('..') / 'dados' / 'clientes' / 'clientes.csv'
NOMES = {'pedidos': 'pedidos_12m', 'gasto': 'gasto_12m', 'ticket': 'ticket_medio',
         'dias': 'dias_desde_ultima_compra', 'categorias': 'categorias_distintas', 'cupom': 'pct_pedidos_cupom',
         'app': 'pct_pedidos_app', 'devolucoes': 'devolucoes_12m'}
COM_LOG = ['pedidos', 'gasto', 'ticket', 'devolucoes']
AGOSTO = {'colunas': 'pedidos,ticket,dias,categorias,cupom,app', 'escala': 'log', 'topo': 40, 'k': 4, 'comecos': 10}
FIRME = 0.9


class Erro(Exception):
    pass


def br(x, casas=1):
    texto = f'{float(x):,.{casas}f}'.replace(',', '_').replace('.', ',').replace('_', '.')
    return texto[1:] if texto.startswith('-') and set(texto[1:]) <= set('0,.') else texto


def ler_argumentos(argv):
    p = argparse.ArgumentParser(description='Ver se os grupos de agosto se mantêm e montar o teste da campanha.')
    modos = p.add_subparsers(dest='modo', required=True, metavar='partida|amostra|comparar|teste')

    def opcoes(m):
        m.add_argument('--colunas', default=AGOSTO['colunas'],
                       help='as colunas da distância (padrão: as seis de agosto)')
        m.add_argument('--escala', choices=['z', 'log'], default=AGOSTO['escala'],
                       help='z, ou log nos pedidos, gasto, ticket e devoluções e depois z (padrão log)')
        m.add_argument('--topo', type=int, default=AGOSTO['topo'],
                       help='separa antes de agrupar quem tem esse número de pedidos ou mais; 0 não separa (padrão 40)')
        m.add_argument('--k', type=int, default=AGOSTO['k'], help='quantos grupos, de 2 a 8 (padrão 4)')
        m.add_argument('--comecos', type=int, default=AGOSTO['comecos'],
                       help='quantos começos o K-means testa em cada rodada, de 1 a 100 (padrão 10)')

    m = modos.add_parser('partida', help='o mesmo K-means com outros pontos de partida')
    opcoes(m)
    m.add_argument('--vezes', type=int, default=20, help='quantas partidas, de 2 a 50 (padrão 20)')
    m = modos.add_parser('amostra', help='o K-means refeito em sorteios de parte dos clientes')
    opcoes(m)
    m.add_argument('--vezes', type=int, default=20, help='quantos sorteios, de 2 a 50 (padrão 20)')
    m.add_argument('--fracao', type=int, default=80, help='porcentagem de clientes em cada sorteio, de 50 a 95 (padrão 80)')
    m.add_argument('--ver', type=int, metavar='N', help='abre o sorteio N: tabela cruzada, medianas e soma das distâncias com 10 e 50 começos')
    m.add_argument('--fronteira', action='store_true', help='quem são os clientes que trocam de grupo com frequência')
    m.add_argument('--salvar', metavar='ARQUIVO.csv', help='grava id_cliente, grupo e firmeza de cada cliente')
    m = modos.add_parser('comparar', help='uma segmentação com outra escolha contra a de agosto')
    opcoes(m)
    m = modos.add_parser('teste', help='divide cada grupo por sorteio para medir a campanha')
    opcoes(m)
    m.add_argument('--sem-email', type=int, default=10, help='porcentagem de cada grupo sem e-mail (padrão 10)')
    m.add_argument('--padrao', type=int, default=10, help='porcentagem de cada grupo com o e-mail padrão (padrão 10)')
    m.add_argument('--taxa', type=int, default=20,
                   help='porcentagem de clientes que você imagina que voltam a comprar, para a margem (padrão 20)')
    m.add_argument('--salvar', metavar='ARQUIVO.csv', help='grava id_cliente, grupo e o e-mail de cada cliente')
    return p.parse_args(argv)


def validar(a):
    nomes = [n.strip() for n in a.colunas.split(',') if n.strip()]
    for n in nomes:
        if n not in NOMES:
            raise Erro(f'não conheço a coluna "{n}". Use: {", ".join(NOMES)}')
    if len(set(nomes)) != len(nomes):
        raise Erro('coluna repetida em --colunas')
    if len(nomes) < 2:
        raise Erro('use pelo menos 2 colunas em --colunas')
    if not 2 <= a.k <= 8:
        raise Erro('--k precisa ficar entre 2 e 8')
    if not 1 <= a.comecos <= 100:
        raise Erro('--comecos precisa ficar entre 1 e 100')
    if a.topo < 0:
        raise Erro('--topo precisa ser 0 (não separa) ou um número de pedidos')
    if a.modo in ('partida', 'amostra') and not 2 <= a.vezes <= 50:
        raise Erro('--vezes precisa ficar entre 2 e 50')
    if a.modo == 'amostra':
        if not 50 <= a.fracao <= 95:
            raise Erro('--fracao precisa ficar entre 50 e 95')
        if a.ver is not None and not 1 <= a.ver <= a.vezes:
            raise Erro(f'--ver precisa ser um sorteio de 1 a {a.vezes}')
    if a.modo == 'teste':
        for opcao, valor in (('--sem-email', a.sem_email), ('--padrao', a.padrao)):
            if not 5 <= valor <= 30:
                raise Erro(f'{opcao} precisa ficar entre 5 e 30')
        if not 1 <= a.taxa <= 99:
            raise Erro('--taxa precisa ficar entre 1 e 99')
    if getattr(a, 'salvar', None) and not a.salvar.endswith('.csv'):
        raise Erro('--salvar precisa de um nome que termine em .csv')
    return nomes


def ler_clientes():
    if not ARQUIVO.is_file():
        raise Erro(f'não achei {ARQUIVO}. Rode de dentro da pasta da aula, 10-estabilidade-e-acao.')
    return pd.read_csv(ARQUIVO)


def separar_topo(base, topo):
    if topo == 0:
        return base.reset_index(drop=True), base.iloc[0:0]
    no_topo = base['pedidos_12m'] >= topo
    return base[~no_topo].reset_index(drop=True), base[no_topo]


def representar(parte, nomes, escala):
    """Log quando pedido e escala z calculada com os clientes da própria rodada."""
    x = pd.DataFrame({n: parte[NOMES[n]].astype(float) for n in nomes})
    if escala == 'log':
        for n in nomes:
            if n in COM_LOG:
                x[n] = np.log1p(x[n])
    m = x.to_numpy(float)
    desvio = m.std(axis=0, ddof=1)
    if (desvio == 0).any():
        raise Erro('uma das colunas tem o mesmo valor para todos os clientes da rodada')
    return (m - m.mean(axis=0)) / desvio


def agrupar(x, k, comecos, partida=0):
    """Retorna os grupos numerados do maior para o menor, a partir de 1, e a soma das distâncias ao quadrado."""
    modelo = KMeans(n_clusters=k, n_init=comecos, random_state=partida).fit(x)
    tamanhos = np.bincount(modelo.labels_, minlength=k)
    ordem = sorted(range(k), key=lambda g: (-tamanhos[g], g))
    mapa = np.empty(k, dtype=int)
    for n, g in enumerate(ordem):
        mapa[g] = n + 1
    return mapa[modelo.labels_], float(modelo.inertia_)


def casar(antes, agora):
    """Casa cada grupo de antes com o grupo de agora que mais divide clientes com ele."""
    tabela = pd.crosstab(pd.Series(antes, name='antes'), pd.Series(agora, name='agora'))
    linhas, colunas = linear_sum_assignment(-tabela.to_numpy())
    casado = {int(tabela.index[i]): int(tabela.columns[j]) for i, j in zip(linhas, colunas)}
    ficou = np.array([casado.get(int(a)) == int(b) for a, b in zip(antes, agora)])
    jaccard = {}
    for g, h in casado.items():
        junto = int(((antes == g) & (agora == h)).sum())
        jaccard[g] = junto / int(((antes == g) | (agora == h)).sum())
    for g in set(int(v) for v in antes) - set(casado):
        jaccard[g] = 0.0
    return tabela, casado, ficou, jaccard


def titulo(texto):
    print(texto)
    print('-' * len(texto))


def descrever(a, nomes, n_topo, n_agrupados):
    diferente = [f'{o} {getattr(a, o)}' for o in ('k', 'escala', 'topo', 'comecos') if getattr(a, o) != AGOSTO[o]]
    if ','.join(nomes) != AGOSTO['colunas']:
        diferente.insert(0, f'colunas {",".join(nomes)}')
    titulo('Segmentação de agosto' + (f', com {"; ".join(diferente)}' if diferente else ''))
    print(f'colunas na distância: {", ".join(nomes)}')
    if a.escala == 'log':
        com = [n for n in nomes if n in COM_LOG]
        print(f'escala: log em {", ".join(com) if com else "nenhuma das colunas escolhidas"}, depois z')
    else:
        print('escala: z')
    if a.topo:
        print(f'clientes de {a.topo} pedidos ou mais: {n_topo} separados antes de agrupar')
    else:
        print('clientes do topo: agrupados com os outros')
    print(f'K-means com K = {a.k} e {a.comecos} começo{"s" if a.comecos > 1 else ""} em cada rodada')
    print(f'clientes agrupados: {n_agrupados}')


def mostrar_tamanhos(rotulos):
    print('tamanhos dos grupos: ' + ', '.join(f'{g}: {int((rotulos == g).sum())}' for g in sorted(set(rotulos))))


def modo_partida(a, x):
    ref, soma_ref = agrupar(x, a.k, a.comecos, 0)
    print()
    mostrar_tamanhos(ref)
    print()
    titulo(f'O mesmo K-means com {a.vezes} pontos de partida (a partida 0 é a de agosto)')
    print(f'{"partida":>8}{"trocam":>9}{"soma das distâncias":>22}')
    print(f'{0:>8}{"":>9}{br(soma_ref, 1):>22}')
    trocas = []
    for partida in range(1, a.vezes):
        rot, soma = agrupar(x, a.k, a.comecos, partida)
        _, _, ficou, _ = casar(ref, rot)
        trocas.append(int((~ficou).sum()))
        print(f'{partida:>8}{trocas[-1]:>9}{br(soma, 1):>22}')
    n = len(x)
    print(f'pior partida: {max(trocas)} de {n} clientes trocam de grupo ({br(100 * max(trocas) / n)}%)')
    print(f'partidas com 10 trocas ou menos: {sum(t <= 10 for t in trocas)} de {len(trocas)}')


def sortear(n, fracao, numero):
    """O sorteio número N sai igual toda vez."""
    rng = np.random.default_rng(1000 + numero)
    return np.sort(rng.choice(n, int(round(n * fracao / 100)), replace=False))


def medianas(parte, rotulos):
    linhas = []
    for g in sorted(set(rotulos)):
        p = parte[rotulos == g]
        linhas.append({'grupo': g, 'clientes': len(p), 'pedidos': br(p['pedidos_12m'].median()),
                       'ticket R$': br(p['ticket_medio'].median(), 0),
                       'dias': br(p['dias_desde_ultima_compra'].median(), 0),
                       'cupom %': br(p['pct_pedidos_cupom'].median(), 0),
                       'app %': br(p['pct_pedidos_app'].median(), 0)})
    return pd.DataFrame(linhas).set_index('grupo').to_string()


def modo_amostra(a, base, nomes, x):
    ref, _ = agrupar(x, a.k, a.comecos, 0)
    n = len(base)
    vezes_dentro = np.zeros(n, dtype=int)
    vezes_ficou = np.zeros(n, dtype=int)
    jaccards, trocas = [], []
    print()
    mostrar_tamanhos(ref)
    print()
    titulo(f'{a.vezes} sorteios de {a.fracao}% dos clientes; em cada um, a escala e o K-means são refeitos')
    print(f'{"sorteio":>8}{"clientes":>10}{"trocam":>8}{"% trocam":>10}{"ARI":>8}')
    for numero in range(1, a.vezes + 1):
        idx = sortear(n, a.fracao, numero)
        rot, soma = agrupar(representar(base.iloc[idx], nomes, a.escala), a.k, a.comecos, 0)
        tabela, casado, ficou, jac = casar(ref[idx], rot)
        vezes_dentro[idx] += 1
        vezes_ficou[idx] += ficou
        jaccards.append(jac)
        trocas.append(int((~ficou).sum()))
        print(f'{numero:>8}{len(idx):>10}{trocas[-1]:>8}{br(100 * trocas[-1] / len(idx)):>10}'
              f'{br(adjusted_rand_score(ref[idx], rot), 3):>8}')
        if a.ver == numero:
            guardado = (idx, rot, soma, tabela, casado)
    pior = int(np.argmax(trocas))
    print(f'pior sorteio: {pior + 1}, com {trocas[pior]} clientes trocando de grupo')
    print(f'sorteios com 1% dos clientes trocando ou menos: {sum(t <= 0.01 * n * a.fracao / 100 for t in trocas)} de {a.vezes}')
    print()
    titulo('Cada grupo de agosto nos sorteios (Jaccard com o grupo casado: 1 é igual, 0 sumiu)')
    jac = pd.DataFrame(jaccards)
    print(f'{"grupo":>6}{"clientes":>10}{"menor":>8}{"mediana":>9}{"abaixo de 0,75":>16}')
    for g in sorted(jac.columns):
        print(f'{g:>6}{int((ref == g).sum()):>10}{br(jac[g].min(), 2):>8}{br(jac[g].median(), 2):>9}'
              f'{int((jac[g] < 0.75).sum()):>16}')
    firmeza = vezes_ficou / np.maximum(vezes_dentro, 1)
    print()
    titulo('Cada cliente nos sorteios em que ele entrou')
    print(f'entrou em pelo menos um sorteio: {int((vezes_dentro > 0).sum())} de {n}')
    print(f'ficou no grupo de agosto em todos: {int(((firmeza == 1) & (vezes_dentro > 0)).sum())}')
    fronteira = (firmeza < FIRME) & (vezes_dentro > 0)
    print(f'ficou em menos de {br(100 * FIRME, 0)}% deles: {int(fronteira.sum())}')
    if a.ver is not None:
        idx, rot, soma, tabela, casado = guardado
        print()
        titulo(f'Sorteio {a.ver}: linhas são os grupos de agosto, colunas os grupos do sorteio')
        print(tabela.to_string())
        print('casamento: ' + ', '.join(f'{g} com {h}' for g, h in sorted(casado.items())))
        print()
        print('Medianas dos grupos do sorteio')
        print(medianas(base.iloc[idx].reset_index(drop=True), rot))
        xs = representar(base.iloc[idx], nomes, a.escala)
        for comecos in sorted({a.comecos, 10, 50}):
            s = soma if comecos == a.comecos else agrupar(xs, a.k, comecos, 0)[1]
            print(f'soma das distâncias no sorteio, com {comecos} começo{"s" if comecos > 1 else ""}: {br(s, 1)}')
    if a.fronteira:
        print()
        titulo(f'Os {int(fronteira.sum())} clientes que ficaram no grupo em menos de {br(100 * FIRME, 0)}% dos sorteios')
        if fronteira.any():
            quem = base[fronteira]
            print('grupo de agosto: ' + ', '.join(f'{g}: {int((ref[fronteira] == g).sum())}' for g in sorted(set(ref[fronteira]))))
            print('mediana: ' + '; '.join(
                f'{rotulo} {br(quem[c].median(), casas)}' for rotulo, c, casas in
                [('pedidos', 'pedidos_12m', 1), ('ticket R$', 'ticket_medio', 0), ('dias sem comprar', 'dias_desde_ultima_compra', 0),
                 ('cupom %', 'pct_pedidos_cupom', 0), ('app %', 'pct_pedidos_app', 0)]))
            print(f'com 1 ou 2 pedidos: {int((quem["pedidos_12m"] <= 2).sum())}')
        else:
            print('nenhum')
    if a.salvar:
        saida = pd.DataFrame({'id_cliente': base['id_cliente'], 'grupo': ref, 'firmeza': np.round(firmeza, 3)})
        saida.to_csv(a.salvar, index=False)
        print(f'\ngravei {len(saida)} clientes em {a.salvar}')


def modo_comparar(a, base_toda, nomes):
    agosto = argparse.Namespace(**AGOSTO)
    base_a, topo_a = separar_topo(base_toda, agosto.topo)
    ref, _ = agrupar(representar(base_a, AGOSTO['colunas'].split(','), agosto.escala), agosto.k, agosto.comecos, 0)
    base_b, topo_b = separar_topo(base_toda, a.topo)
    rot, _ = agrupar(representar(base_b, nomes, a.escala), a.k, a.comecos, 0)
    antes = pd.Series(ref.astype(str), index=base_a['id_cliente'])
    antes = pd.concat([antes, pd.Series('topo', index=topo_a['id_cliente'])])
    agora = pd.Series(rot.astype(str), index=base_b['id_cliente'])
    agora = pd.concat([agora, pd.Series('topo', index=topo_b['id_cliente'])])
    agora = agora.loc[antes.index]
    tabela = pd.crosstab(antes.rename('agosto'), agora.rename('agora'))
    linhas, colunas = linear_sum_assignment(-tabela.to_numpy())
    mesmos = int(tabela.to_numpy()[linhas, colunas].sum())
    print()
    titulo('Linhas: grupos de agosto; colunas: grupos com a escolha nova')
    print(tabela.to_string())
    print('casamento: ' + ', '.join(f'{tabela.index[i]} com {tabela.columns[j]}' for i, j in zip(linhas, colunas)))
    n = len(antes)
    print(f'clientes que trocam de grupo: {n - mesmos} de {n} ({br(100 * (n - mesmos) / n)}%)')
    print(f'ARI contra agosto: {br(adjusted_rand_score(antes.to_numpy(), agora.to_numpy()), 3)}')
    print()
    print('Medianas dos grupos com a escolha nova')
    print(medianas(base_b, rot))


def margem(p, n1, n2):
    return 196 * np.sqrt(p * (1 - p) / n1 + p * (1 - p) / n2)


def modo_teste(a, base, x, topo):
    ref, _ = agrupar(x, a.k, a.comecos, 0)
    rng = np.random.default_rng(2026)
    p = a.taxa / 100
    braco = np.empty(len(base), dtype=object)
    print()
    titulo(f'Teste de agosto: em cada grupo, {a.sem_email}% sem e-mail e {a.padrao}% com o e-mail padrão, por sorteio')
    print(f'{"grupo":>6}{"clientes":>10}{"do grupo":>10}{"padrão":>8}{"sem e-mail":>12}'
          f'{"margem grupo x padrão":>23}{"margem grupo x nada":>21}')
    totais = {'grupo': 0, 'padrao': 0, 'nada': 0}
    for g in sorted(set(ref)):
        quem = np.flatnonzero(ref == g)
        rng.shuffle(quem)
        n_nada = int(round(len(quem) * a.sem_email / 100))
        n_padrao = int(round(len(quem) * a.padrao / 100))
        braco[quem[:n_nada]] = 'sem e-mail'
        braco[quem[n_nada:n_nada + n_padrao]] = 'padrão'
        braco[quem[n_nada + n_padrao:]] = f'grupo {g}'
        n_grupo = len(quem) - n_nada - n_padrao
        totais['grupo'] += n_grupo
        totais['padrao'] += n_padrao
        totais['nada'] += n_nada
        print(f'{g:>6}{len(quem):>10}{n_grupo:>10}{n_padrao:>8}{n_nada:>12}'
              f'{"± " + br(margem(p, n_grupo, n_padrao)) + " pontos":>23}{"± " + br(margem(p, n_grupo, n_nada)) + " pontos":>21}')
    print(f'{"todos":>6}{len(base):>10}{totais["grupo"]:>10}{totais["padrao"]:>8}{totais["nada"]:>12}'
          f'{"± " + br(margem(p, totais["grupo"], totais["padrao"])) + " pontos":>23}'
          f'{"± " + br(margem(p, totais["grupo"], totais["nada"])) + " pontos":>21}')
    print(f'margem de 95% para a diferença entre duas taxas perto de {a.taxa}%; diferença menor que a margem não dá para separar do acaso')
    if len(topo):
        print(f'fora do teste: {len(topo)} clientes de {a.topo} pedidos ou mais, na lista do comercial')
    if a.salvar:
        saida = pd.DataFrame({'id_cliente': base['id_cliente'], 'grupo': ref.astype(str), 'email': braco})
        if len(topo):
            saida = pd.concat([saida, pd.DataFrame({'id_cliente': topo['id_cliente'], 'grupo': 'topo', 'email': 'comercial'})])
        saida.sort_values('id_cliente').to_csv(a.salvar, index=False)
        print(f'\ngravei {len(saida)} clientes em {a.salvar}')


def main(argv):
    a = ler_argumentos(argv)
    try:
        nomes = validar(a)
        base_toda = ler_clientes()
        base, topo = separar_topo(base_toda, a.topo)
        descrever(a, nomes, len(topo), len(base))
        if a.modo == 'comparar':
            modo_comparar(a, base_toda, nomes)
            return 0
        x = representar(base, nomes, a.escala)
        if a.modo == 'partida':
            modo_partida(a, x)
        elif a.modo == 'amostra':
            modo_amostra(a, base, nomes, x)
        else:
            modo_teste(a, base, x, topo)
    except Erro as e:
        print(f'erro: {e}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
