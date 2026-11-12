"""As contas da aula «Autovalores, autovetores e SVD», na ordem em que a aula mostra.

    python3 transformacao.py

1. A matriz que estica o plano, a mesma da tarefa 3 do diagnóstico.
2. Uma matriz que gira: nenhuma direção fica no lugar, e o SVD mostra o que ela faz.
3. Pedidos e gasto dos clientes: a pergunta da gerente.
"""
import sys

import numpy as np
import pandas as pd

from duas_colunas import ARQUIVO, matriz
from ver_matriz import num, relatorio, vet


def titulo(texto):
    print()
    print(texto)
    print('-' * len(texto))


def parte_clientes():
    clientes = pd.read_csv(ARQUIVO)
    n, c = matriz(clientes, 'pedidos_12m', 'gasto_12m')
    print(f'{n} clientes; x = pedidos_12m, y = gasto_12m, cada uma em escala z')
    print('matriz de covariância')
    for linha in c.values:
        print('  | ' + '  '.join(num(v, 4) for v in linha) + ' |')
    valores, vetores = np.linalg.eigh(c.values)
    print('np.linalg.eigh, como sai:')
    print('  autovalores ' + '  '.join(num(v, 4) for v in valores))
    print('  autovetores (uma coluna para cada autovalor)')
    for linha in vetores:
        print('    ' + '  '.join(num(v, 4) for v in linha))
    maior = valores[-1]
    print(f'direção mais comprida {vet(vetores[:, -1] * np.sign(vetores[0, -1]))}, '
          f'com {num(maior, 4)} de {num(valores.sum(), 4)}: {num(100 * maior / valores.sum(), 1)}% da variância')
    par = clientes[['pedidos_12m', 'gasto_12m']]
    z = ((par - par.mean()) / par.std()).values
    s = np.linalg.svd(z, compute_uv=False)
    print('SVD da tabela em escala z (3000 linhas x 2 colunas)')
    print('  valores singulares ' + '  '.join(num(v) for v in s))
    print(f'  ao quadrado e divididos por n - 1 = {n - 1}: ' + '  '.join(num(v * v / (n - 1), 4) for v in s))
    sem = clientes.drop(clientes.nlargest(30, 'gasto_12m').index)
    n2, c2 = matriz(sem, 'pedidos_12m', 'gasto_12m')
    v2 = np.linalg.eigvalsh(c2.values)
    print(f'sem os 30 maiores gastos ({n2} clientes): covariância {num(c2.values[0, 1], 4)}, '
          f'autovalores {num(v2[1], 4)} e {num(v2[0], 4)}, a direção mais comprida fica com '
          f'{num(100 * v2[1] / v2.sum(), 1)}%')


def main():
    titulo('1. Uma matriz que estica o plano')
    print(relatorio(np.array([[2.0, 1.0], [1.0, 2.0]])))
    titulo('2. Uma matriz que gira')
    print(relatorio(np.array([[1.0, 1.0], [-2.0, 2.0]])))
    titulo('3. Pedidos e gasto dos clientes')
    if not ARQUIVO.exists():
        print(f'erro: não achei {ARQUIVO}. Rode de dentro da pasta da aula.', file=sys.stderr)
        return 2
    parte_clientes()
    return 0


if __name__ == '__main__':
    sys.exit(main())
