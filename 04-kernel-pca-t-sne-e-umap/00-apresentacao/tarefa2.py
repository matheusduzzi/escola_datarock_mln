"""Tarefa 2 do diagnóstico: padronizar colunas e medir distâncias, sem loop em Python.

Rode de dentro da pasta 00-apresentacao:

    python3 tarefa2.py
    python3 testar_tarefa2.py
"""
import numpy as np


def padronizar(X):
    """Retorna um array do mesmo formato de X, com cada coluna na escala z: (x - média) / desvio padrão.

    Média e desvio padrão são de cada coluna (desvio padrão populacional, o padrão do NumPy).
    X pode chegar como lista de listas. Para com ValueError se alguma coluna tiver todos os valores iguais.
    """
    raise NotImplementedError('escreva padronizar')


def distancias(X):
    """Retorna a matriz n x n das distâncias euclidianas entre as linhas de X."""
    raise NotImplementedError('escreva distancias')


if __name__ == '__main__':
    # três clientes: gasto em reais e número de pedidos
    X = [[100, 1], [140, 1], [100, 5]]
    print('sem escala:', np.round(distancias(X), 3).tolist())
    print('padronizado:', np.round(distancias(padronizar(X)), 3).tolist())
