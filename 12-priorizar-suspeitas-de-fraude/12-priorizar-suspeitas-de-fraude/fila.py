"""Fila de revisão das compras com cartão: as 5 de nota mais alta de cada dia e 1 sorteada.

Rode de dentro da pasta desta aula:

    python3 fila.py

A nota é aprendida com as compras de janeiro a março e aplicada, dia a dia, de abril a junho.
A coluna `fraude` só entra na parte 4, como se a contestação do banco tivesse chegado.
"""
import numpy as np
import pandas as pd

from nota import COLUNAS, ajustar, motivos, pontuar

DADOS = '../dados/fraude/transacoes.csv'
INICIO_FILA = '2026-04-01'
TOPO = 5
SORTEADAS = 1
DIA_EXEMPLO = '2026-04-21'
SORTEIO = 20260401  # o mesmo sorteio toda vez que o programa roda


def br(x, casas=2):
    """Número com vírgula decimal e ponto de milhar, como no resto da aula."""
    texto = f'{x:,.{casas}f}'
    return texto.replace(',', '_').replace('.', ',').replace('_', '.')


def carregar(caminho=DADOS):
    compras = pd.read_csv(caminho)
    compras['dia'] = compras['data_hora'].str[:10]
    return compras


def montar_fila(compras, topo, sorteadas, sorteio, por='nota'):
    """Para cada dia, as `topo` compras de maior `por` e mais `sorteadas` entre as que sobraram.

    Empate na nota: entra primeiro a compra mais antiga (menor id_transacao).
    Retorna as compras escolhidas com duas colunas novas: `origem` (topo ou sorteada) e `posicao`.
    """
    if topo < 0 or sorteadas < 0:
        raise ValueError('topo e sorteadas não podem ser negativos')
    if por not in compras.columns:
        raise ValueError(f'a coluna {por!r} não existe nas compras')
    if compras[por].isna().any():
        raise ValueError(f'há compras sem {por!r}; calcule a nota de todas antes de montar a fila')
    partes = []
    for dia, do_dia in compras.groupby('dia', sort=True):
        ordem = do_dia.sort_values([por, 'id_transacao'], ascending=[False, True])
        cima = ordem.head(topo).assign(origem='topo')
        cima['posicao'] = range(1, len(cima) + 1)
        resto = ordem.iloc[len(cima):]
        k = min(sorteadas, len(resto))
        escolha = sorteio.choice(len(resto), size=k, replace=False) if k else []
        sorteio_dia = resto.iloc[sorted(escolha)].assign(origem='sorteada')
        sorteio_dia['posicao'] = range(len(cima) + 1, len(cima) + len(sorteio_dia) + 1)
        partes += [cima, sorteio_dia]
    return pd.concat(partes) if partes else compras.head(0).assign(origem=[], posicao=[])


def preparar(compras, escala=None):
    """Nota aprendida com janeiro a março, aplicada às compras de abril em diante."""
    referencia = compras[compras['dia'] < INICIO_FILA]
    novas = compras[compras['dia'] >= INICIO_FILA].copy()
    raridade = pontuar(ajustar(referencia, escala=escala), novas)
    novas = novas.join(raridade.add_prefix('r_').rename(columns={'r_nota': 'nota'}))
    return referencia, novas


def main():
    compras = carregar()
    referencia, novas = preparar(compras)
    sem_fraude = novas.drop(columns='fraude')  # a fila é montada sem a resposta
    fila = montar_fila(sem_fraude, TOPO, SORTEADAS, np.random.default_rng(SORTEIO))
    antiga = montar_fila(sem_fraude, TOPO, SORTEADAS, np.random.default_rng(SORTEIO), por='distancia_entrega_km')

    print(f'1. Referência: compras de {referencia.dia.min()} a {referencia.dia.max()}')
    print(f'   compras: {len(referencia)}; faixas por coluna contínua: 10; a coluna fraude não entra na nota')
    m = ajustar(referencia)['colunas']['tentativas_cartao']
    print('   tentativas do cartão: valor, quantas compras, raridade')
    for valor, fracao in m['fracao'].sort_index().items():
        print(f'      {valor}: {int(round(fracao * len(referencia))):>4}  {br(np.log10(m["maior"] / fracao))}')

    dia = novas[novas.dia == DIA_EXEMPLO].sort_values(['nota', 'id_transacao'], ascending=[False, True])
    hoje = fila[fila.dia == DIA_EXEMPLO]
    print(f'2. A fila de {DIA_EXEMPLO} ({len(dia)} compras; a equipe revisa {TOPO + SORTEADAS})')
    print('   pos  id     nota   valor     conta  tent  dist km  hora  pesou mais')
    linhas = list(hoje.itertuples()) + [None] + [list(dia.iloc[[TOPO]].itertuples())[0]]
    for l in linhas:
        if l is None:
            print('   --- fora do orçamento: a 6ª nota do dia segue para a entrega ---')
            continue
        r = novas.loc[l.Index, ['r_' + c for c in COLUNAS]]
        r.index = COLUNAS
        pos = 'S' if getattr(l, 'origem', '') == 'sorteada' else str(int(dia.index.get_loc(l.Index)) + 1)
        print(f'   {pos:>3}  {l.id_transacao:<5}  {br(l.nota):>5}  {br(l.valor):>8}  {l.idade_conta_dias:>5}  '
              f'{l.tentativas_cartao:>4}  {br(l.distancia_entrega_km, 1):>7}  {l.hora:>4}  {", ".join(motivos(r, 3))}')
    print('   S = sorteada entre as que ficaram fora das 5 do topo')

    print('3. Abril a junho: a fila de cada dia')
    print(f'   dias: {novas.dia.nunique()}; compras: {len(novas)}; revisões: {len(fila)} '
          f'({(fila.origem == "topo").sum()} do topo + {(fila.origem == "sorteada").sum()} sorteadas)')
    por_dia = fila.groupby('dia').size()
    faixa = f'{por_dia.min()} em todos os dias' if por_dia.min() == por_dia.max() else f'de {por_dia.min()} a {por_dia.max()}'
    print(f'   revisões por dia: {faixa}')

    print('4. Quando a contestação chega (a coluna fraude só entra aqui)')
    fraude = novas['fraude']
    total = int(fraude.sum())
    print(f'   fraudes de abril a junho: {total}')
    for nome, f in [('fila nova', fila), ('regra antiga (5 entregas mais longe + 1 sorteada)', antiga)]:
        no_topo = int(fraude.loc[f.index[f.origem == 'topo']].sum())
        sorteada = int(fraude.loc[f.index[f.origem == 'sorteada']].sum())
        print(f'   {nome}: {no_topo} no topo, {sorteada} nas sorteadas, {total - no_topo - sorteada} passaram')
        print(f'      compras do topo que não eram fraude: {(f.origem == "topo").sum() - no_topo} '
              f'de {(f.origem == "topo").sum()}')
    print(f'   6 sorteadas por dia, sem nota nenhuma: umas {br(len(fila) * total / len(novas), 1)} fraudes esperadas')

    print('5. Quantas do topo por dia (fila nova, sem contar a sorteada)')
    print('   topo  revisões  fraudes  fraudes a mais')
    ordem = novas.sort_values(['dia', 'nota', 'id_transacao'], ascending=[True, False, True])
    posicao = ordem.groupby('dia').cumcount() + 1
    antes = 0
    for k in range(1, TOPO + 1):
        pegou = int(ordem.loc[posicao <= k, 'fraude'].sum())
        print(f'   {k:>4}  {int((posicao <= k).sum()):>8}  {pegou:>7}  {pegou - antes:>14}')
        antes = pegou

    print('6. A mesma nota com as faixas na escala comum, sem log')
    _, linear = preparar(compras, escala='linear')
    fila_linear = montar_fila(linear.drop(columns='fraude'), TOPO, SORTEADAS, np.random.default_rng(SORTEIO))
    no_topo = int(fraude.loc[fila_linear.index[fila_linear.origem == 'topo']].sum())
    novas_contas = novas[(novas.fraude == 1) & (novas.idade_conta_dias < 30)]
    print(f'   fraudes nas 5 do topo: {no_topo} de {total}')
    print(f'   raridade média da idade da conta nas fraudes com conta de menos de 30 dias: '
          f'{br(linear.loc[novas_contas.index, "r_idade_conta_dias"].mean())} sem log, '
          f'{br(novas.loc[novas_contas.index, "r_idade_conta_dias"].mean())} com log')


if __name__ == '__main__':
    main()
