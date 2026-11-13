"""Teste da tarefa 2. Rode de dentro da pasta 00-apresentacao: python3 testar_tarefa2.py

Sai com 0 e mostra «5 testes passaram» quando as funções estão certas.
"""
import numpy as np

from tarefa2 import distancias, padronizar


def testar_padronizar_por_coluna():
    Z = padronizar([[1, 100], [2, 300], [3, 500]])
    assert isinstance(Z, np.ndarray) and Z.shape == (3, 2), 'padronizar precisa retornar um array do mesmo formato'
    assert np.allclose(Z.mean(axis=0), 0), f'cada coluna precisa ficar com média 0: {Z.mean(axis=0)}'
    assert np.allclose(Z.std(axis=0), 1), f'cada coluna precisa ficar com desvio padrão 1: {Z.std(axis=0)}'
    assert np.allclose(Z[:, 0], Z[:, 1]), 'as duas colunas têm o mesmo desenho e precisam ficar iguais depois da escala'


def testar_coluna_constante():
    try:
        padronizar([[1, 7], [2, 7], [3, 7]])
    except ValueError:
        return
    raise AssertionError('uma coluna com todos os valores iguais precisa parar com ValueError')


def testar_distancia_conhecida():
    D = distancias([[0, 0], [3, 4], [6, 8]])
    assert isinstance(D, np.ndarray) and D.shape == (3, 3), 'distancias precisa retornar uma matriz n x n'
    assert np.allclose(D, [[0, 5, 10], [5, 0, 5], [10, 5, 0]]), f'distâncias erradas: {D.tolist()}'


def testar_simetria_e_diagonal():
    X = np.arange(12, dtype=float).reshape(4, 3) ** 1.5
    D = distancias(X)
    assert np.allclose(D, D.T), 'a distância de a até b é a mesma de b até a'
    assert np.allclose(np.diag(D), 0), 'a distância de um ponto até ele mesmo é 0'


def testar_exemplo_dos_clientes():
    D = distancias(padronizar([[100, 1], [140, 1], [100, 5]]))
    assert np.allclose(D[0, 1], D[0, 2]), 'depois da escala, R$ 40 e 4 pedidos de diferença pesam igual neste exemplo'


if __name__ == '__main__':
    testes = [testar_padronizar_por_coluna, testar_coluna_constante, testar_distancia_conhecida,
              testar_simetria_e_diagonal, testar_exemplo_dos_clientes]
    for t in testes:
        t()
    print(f'{len(testes)} testes passaram')
