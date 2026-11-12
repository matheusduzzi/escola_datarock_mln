"""Tarefa 4 do diagnóstico: o cliente típico e os clientes do topo.

Rode de dentro da pasta 00-apresentacao:

    python3 tarefa4.py
"""
from pathlib import Path

import pandas as pd

clientes = pd.read_csv(Path(__file__).resolve().parent.parent / 'dados/clientes/clientes.csv')
gasto = clientes['gasto_12m']
maiores = clientes.nlargest(30, 'gasto_12m')
resto = clientes.drop(maiores.index)

print('clientes:', len(clientes))
print(f'gasto em 12 meses: média {gasto.mean():.2f}, mediana {gasto.median():.2f}')
print(f'os 30 maiores somam {100 * maiores["gasto_12m"].sum() / gasto.sum():.1f}% do gasto')
print(f'desvio padrão do gasto: {gasto.std():.2f} com os 30 maiores, {resto["gasto_12m"].std():.2f} sem eles')
colunas = ['id_cliente', 'pedidos_12m', 'gasto_12m', 'ticket_medio', 'dias_desde_ultima_compra', 'pct_pedidos_app']
print(maiores[colunas].head(5).to_string(index=False))
