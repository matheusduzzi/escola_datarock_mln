"""Os números que um acordo de uso precisa antes do primeiro modelo.

Rode de dentro da pasta 01-objetivo-sem-rotulo/:

    python3 numeros_do_acordo.py

Não treina modelo nenhum. Conta o que a operação da loja precisa saber para decidir o tamanho
dos grupos, o tamanho da fila e como medir se ajudou. Só usa o pandas.
"""
import pandas as pd

CLIENTES = '../dados/clientes/clientes.csv'
COMPRAS = '../dados/fraude/transacoes.csv'
SEIS = [58, 94, 69, 498, 217, 281]


def br(x, casas=2):
    """Número no jeito brasileiro: 1.234,56."""
    s = f'{x:,.{casas}f}'
    return s.replace(',', '_').replace('.', ',').replace('_', '.')


c = pd.read_csv(CLIENTES)
t = pd.read_csv(COMPRAS, parse_dates=['data_hora'])

print('1. Seis clientes, duas perguntas')
print('   id   pedidos   gasto 12m   dias sem comprar')
seis = c.set_index('id_cliente').loc[SEIS]
for i, l in seis.iterrows():
    print(f'   {i:<4} {l.pedidos_12m:>7}   {br(l.gasto_12m):>9}   {l.dias_desde_ultima_compra:>16}')
gasta = sorted(seis.index[seis.gasto_12m >= 1000].tolist())
sumiu = sorted(seis.index[seis.dias_desde_ultima_compra > 180].tolist())
print(f'   pelo gasto (R$ 1.000 ou mais): {gasta}')
print(f'   pela volta (mais de 180 dias sem comprar): {sumiu}')
print(f'   nos dois: {sorted(set(gasta) & set(sumiu))}')

print('2. Clientes para a campanha de agosto (base até 2026-06-30)')
print(f'   clientes: {len(c)}')
print(f'   compraram nos últimos 90 dias: {(c.dias_desde_ultima_compra <= 90).sum()}')
print(f'   mais de 180 dias sem comprar: {(c.dias_desde_ultima_compra > 180).sum()}')
print(f'   um pedido só no ano: {(c.pedidos_12m == 1).sum()}')
print(f'   nunca avaliaram um produto (avaliação vazia): {c.avaliacao_media.isna().sum()}')

print('3. Compras com cartão por dia (2026-01-01 a 2026-06-30)')
dia = t.groupby(t.data_hora.dt.date).agg(compras=('fraude', 'size'), fraudes=('fraude', 'sum'))
print(f'   dias: {len(dia)}; compras: {len(t)}; fraudes contestadas depois: {int(t.fraude.sum())}')
print(f'   compras por dia: mediana {int(dia.compras.median())}, de {dia.compras.min()} a {dia.compras.max()}')
print(f'   fraudes por dia: média {br(dia.fraudes.mean())}; dias sem nenhuma: {(dia.fraudes == 0).sum()}')
revisa = 6
print(f'   a equipe revisa {revisa} por dia: {br(100 * revisa / dia.compras.median(), 1)}% de um dia mediano')
sorteadas = len(dia)
esperado = sorteadas * t.fraude.sum() / len(t)
print(f'   1 compra sorteada fora da fila por dia: {sorteadas} no semestre;'
      f' se a fila não pegasse nenhuma fraude, elas achariam umas {br(esperado, 1)}')

print('4. Devoluções dos clientes')
taxa = c.devolucoes_12m / c.pedidos_12m
print(f'   com pelo menos uma devolução: {(c.devolucoes_12m > 0).sum()}')
print(f'   com 3 devoluções ou mais: {(c.devolucoes_12m >= 3).sum()}')
print(f'   devolveram todos os pedidos: {(taxa == 1).sum()}'
      f' (com um pedido só: {((c.pedidos_12m == 1) & (c.devolucoes_12m == 1)).sum()})')
print(f'   devolveram metade dos pedidos ou mais: {(taxa >= 0.5).sum()}')
print(f'   com devolução e avaliação vazia: {((c.devolucoes_12m > 0) & c.avaliacao_media.isna()).sum()}')
top = c.sort_values(['devolucoes_12m', 'id_cliente'], ascending=[False, True]).head(5)
print('   os 5 com mais devoluções (id: devoluções de pedidos):')
print('   ' + '; '.join(f'{l.id_cliente}: {l.devolucoes_12m} de {l.pedidos_12m}' for l in top.itertuples()))
