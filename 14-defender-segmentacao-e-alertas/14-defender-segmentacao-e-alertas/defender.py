"""Os números do relatório: os grupos de clientes da Sônia e a fila de revisão do Marcos.

Rode de dentro da pasta desta aula:

    python3 defender.py

Os limites da campanha e da fila entram como opções. Os valores padrão são os de agosto:

    python3 defender.py --emails 4 --minimo 200 --k 4 --topo 5 --sorteadas 1

A coluna `fraude` não entra na nota nem na fila. Ela só aparece nas partes 5 e 6, como se as
contestações de abril a junho já tivessem chegado.
"""
import argparse

import numpy as np
import pandas as pd

from alertas import carregar, com_nota, montar_fila, wilson
from segmentos import cabe, estabilidade, kmeans, preparar, silhouette, sortear_sem_email, trocam

CLIENTES = '../dados/clientes/clientes.csv'
COLUNAS = ['dias_desde_ultima_compra', 'pedidos_12m', 'gasto_12m', 'pct_pedidos_cupom']
LOG = ['dias_desde_ultima_compra', 'pedidos_12m', 'gasto_12m']
PARTIDAS = 20      # a partida 0 e mais 19, para o mesmo K
RODADAS = 20       # amostras de 80% dos clientes
SEM_EMAIL = 0.10   # parte de cada grupo sorteada para não receber e-mail
PARTIDA_SEM_EMAIL = 14
REVENDA = 50       # pedidos no ano a partir dos quais a compra parece de revenda


def br(x, casas=1):
    texto = f'{x:,.{casas}f}'
    return texto.replace(',', '_').replace('.', ',').replace('_', '.')


def pct(x, casas=1):
    return br(100 * x, casas) + '%'


def grupos(args):
    clientes = pd.read_csv(CLIENTES)
    Z = preparar(clientes, COLUNAS, LOG)

    print(f'1. Quantos grupos: K de 2 a 6 ({len(clientes)} clientes; {len(COLUNAS)} colunas; log em {len(LOG)}; escala z)')
    print(f'   a campanha: até {args.emails} e-mails, cada grupo com pelo menos {args.minimo} clientes')
    print('   K  tamanhos                     silhouette  outra partida  outra amostra  cabe')
    for k in range(2, 7):
        trocas, _, base = estabilidade(Z, k, RODADAS, 0.8, 0)
        pior_partida = max(trocam(base, kmeans(Z, k, p)[0], k) for p in range(1, PARTIDAS))
        tamanhos = np.bincount(base, minlength=k)
        print(f'   {k}  {" ".join(str(t) for t in tamanhos):<27}  {br(silhouette(Z, base), 3):>10}  '
              f'{pior_partida:>13}  {pct(trocas.max()):>13}  {"sim" if cabe(tamanhos, args.emails, args.minimo) else "não"}')
    print(f'   outra partida: no pior de {PARTIDAS - 1} outros pontos de partida, quantos clientes trocam de grupo')
    print(f'   outra amostra: no pior de {RODADAS} sorteios de 80% dos clientes, quantos trocam')

    k = args.k
    trocas, ficam, base = estabilidade(Z, k, RODADAS, 0.8, 0)
    print(f'2. Os {k} grupos (medianas; o grupo 1 é o maior)')
    print('   grupo  clientes  dias sem comprar  pedidos   gasto 12m    ticket  % cupom')
    for j in range(k):
        g = clientes[base == j]
        print(f'   {j + 1:>5}  {len(g):>8}  {g.dias_desde_ultima_compra.median():>16.0f}  {g.pedidos_12m.median():>7.0f}  '
              f'{br(g.gasto_12m.median(), 2):>10}  {br(g.ticket_medio.median(), 2):>8}  {g.pct_pedidos_cupom.median():>7.0f}')
    if k >= 3:
        menos, _ = kmeans(Z, k - 1, 0)
        juntos = pd.crosstab(base, menos)
        print(f'   com um grupo a menos (K = {k - 1}), para onde vai cada grupo: ' + '; '.join(
            f'grupo {j + 1}: ' + ', '.join(f'{n} no {i + 1}' for i, n in juntos.loc[j].items() if n)
            for j in range(k)))
    revenda = clientes.pedidos_12m >= REVENDA
    onde = np.bincount(base[revenda], minlength=k)
    print(f'   clientes com {REVENDA} pedidos ou mais no ano: {int(revenda.sum())}, '
          + ', '.join(f'{n} no grupo {j + 1}' for j, n in enumerate(onde) if n)
          + f'; mediana de {clientes.pedidos_12m[revenda].median():.0f} pedidos e R$ {br(clientes.gasto_12m[revenda].median(), 2)}')

    print(f'3. Quanto os {k} grupos se mantêm')
    print(f'   em {RODADAS} amostras de 80%: clientes que trocam de grupo, de {pct(trocas.min())} a {pct(trocas.max())}')
    print('   parte de cada grupo que fica junta, no pior sorteio: '
          + '; '.join(f'grupo {j + 1} {pct(ficam[:, j].min(), 0)}' for j in range(k)))
    print('   mudando as colunas (comparado com os grupos da parte 2):')
    variantes = [
        ('sem log', COLUNAS, []),
        ('mais pct_pedidos_app', COLUNAS + ['pct_pedidos_app'], LOG),
        ('mais categorias_distintas', COLUNAS + ['categorias_distintas'], LOG),
        ('sem pct_pedidos_cupom', COLUNAS[:3], LOG),
    ]
    for nome, cols, lg in variantes:
        g, _ = kmeans(preparar(clientes, cols, lg), k, 0)
        n = trocam(base, g, k)
        print(f'      {nome:<26} tamanhos {" ".join(str(t) for t in np.bincount(g, minlength=k)):<20} '
              f'trocam {n:>4} ({pct(n / len(clientes))})')

    fora = sortear_sem_email(base, SEM_EMAIL, PARTIDA_SEM_EMAIL)
    print(f'4. A parte sorteada que não recebe e-mail ({pct(SEM_EMAIL, 0)} de cada grupo)')
    print('   ' + '; '.join(f'grupo {j + 1}: {int(fora[base == j].sum())} de {int((base == j).sum())}' for j in range(k)))
    menor = int(min(fora[base == j].sum() for j in range(k)))
    de, ate = wilson(round(menor / 5), menor)
    print(f'   no grupo com menos sorteados ({menor}), se 1 em cada 5 voltar a comprar, a margem de 95% vai de '
          f'{pct(de)} a {pct(ate)}')


def fila(args):
    compras = carregar()
    novas = com_nota(compras)
    resposta = novas['fraude']
    sem_resposta = novas.drop(columns='fraude')
    t, s = args.topo, args.sorteadas
    nova = montar_fila(sem_resposta, t, s)
    regra = montar_fila(sem_resposta, t, s, por='distancia_entrega_km')
    topo = nova.index[nova.origem == 'topo']
    sorteadas = nova.index[nova.origem == 'sorteada']
    total = int(resposta.sum())

    print(f'5. A fila de abril a junho: {t} do topo + {s} {"sorteada" if s == 1 else "sorteadas"} por dia '
          '(a contestação entra só para contar)')
    print(f'   dias: {novas.dia.nunique()}; compras: {len(novas)}; revisões: {len(nova)}; fraudes no trimestre: {total}')
    for nome, f in [('fila nova', nova), ('regra antiga (entrega mais longe)', regra)]:
        no_topo = int(resposta.loc[f.index[f.origem == 'topo']].sum())
        n_topo = int((f.origem == 'topo').sum())
        print(f'   {nome:<34} fraudes no topo: {no_topo:>2} de {n_topo}; alertas úteis {pct(no_topo / n_topo)}; '
              f'compras paradas e liberadas: {n_topo - no_topo}')
    print('   fraudes no topo por mês (fila nova): ' + ', '.join(
        f'{nome} {int(resposta.loc[topo][novas.loc[topo, "dia"].str[5:7] == m].sum())}'
        f' de {int(resposta[novas.dia.str[5:7] == m].sum())}'
        for m, nome in [('04', 'abril'), ('05', 'maio'), ('06', 'junho')]))

    print('6. Quanto a fila se mantém, com o mesmo orçamento')
    print(f'   {"variação":<38} {"compras do topo iguais":<23} fraudes no topo')
    base = set(topo)
    variantes = [('20 faixas em vez de 10', dict(faixas=20)), ('5 faixas', dict(faixas=5)),
                 ('normal aprendido em fevereiro e março', dict(inicio_ref='2026-02-01')),
                 ('normal aprendido só em março', dict(inicio_ref='2026-03-01')),
                 ('faixas sem log', dict(escala='linear'))]
    for nome, kw in variantes:
        v = com_nota(compras, **kw)
        fv = montar_fila(v.drop(columns='fraude'), t, s)
        tv = fv.index[fv.origem == 'topo']
        iguais = len(base & set(tv))
        print(f'   {nome:<38} {iguais:>4} de {len(tv)} ({pct(iguais / len(tv), 0):>4})   '
              f'{int(resposta.loc[tv].sum()):>2} de {total}')

    print('7. O que as sorteadas mostram sem a contestação')
    achadas = int(resposta.loc[sorteadas].sum())
    fora_topo = len(novas) - len(topo)
    if len(sorteadas):
        de, ate = wilson(achadas, len(sorteadas))
        print(f'   sorteadas: {len(sorteadas)}; fraudes nelas: {achadas}; compras fora do topo: {fora_topo}')
        print(f'   margem de 95% (Wilson) para a taxa fora do topo: de {pct(de, 2)} a {pct(ate, 2)}, '
              f'ou de {round(de * fora_topo)} a {round(ate * fora_topo)} fraudes')
    else:
        print(f'   nenhuma sorteada: as {fora_topo} compras fora do topo não têm medida nenhuma')


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--emails', type=int, default=4, help='quantos e-mails a campanha comporta')
    p.add_argument('--minimo', type=int, default=200, help='menor grupo que vale um e-mail')
    p.add_argument('--k', type=int, default=4, help='quantos grupos detalhar nas partes 2 a 4')
    p.add_argument('--topo', type=int, default=5, help='revisões do topo por dia')
    p.add_argument('--sorteadas', type=int, default=1, help='revisões sorteadas por dia')
    args = p.parse_args()
    if not 2 <= args.k <= 6:
        p.error('--k fica entre 2 e 6')
    grupos(args)
    fila(args)


if __name__ == '__main__':
    main()
