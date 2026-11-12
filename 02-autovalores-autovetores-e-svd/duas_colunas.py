"""Monta a matriz 2 x 2 de covariância de duas colunas dos clientes, em escala z.

    python3 duas_colunas.py pedidos_12m gasto_12m
    python3 duas_colunas.py pedidos_12m gasto_12m --sem-maiores-gastos 30

Ele só monta a matriz. As direções e quanto cada uma estica você calcula à mão e depois
vê se bateu com o ver_matriz.py.
"""
import sys
from pathlib import Path

import pandas as pd

ARQUIVO = Path('..') / 'dados' / 'clientes' / 'clientes.csv'
FORA = {'id_cliente', 'data_cadastro', 'regiao'}


def matriz(clientes, a, b):
    """Covariância das duas colunas depois da escala z: é a matriz de correlação."""
    par = clientes[[a, b]].dropna()
    z = (par - par.mean()) / par.std()
    return len(par), z.cov()


def main(argv):
    args = list(argv)
    tirar = 0
    if '--sem-maiores-gastos' in args:
        i = args.index('--sem-maiores-gastos')
        try:
            tirar = int(args[i + 1])
        except (IndexError, ValueError):
            print('erro: depois de --sem-maiores-gastos vem quantos clientes tirar, por exemplo 30', file=sys.stderr)
            return 2
        del args[i:i + 2]
    if not ARQUIVO.exists():
        print(f'erro: não achei {ARQUIVO}. Rode de dentro da pasta da aula.', file=sys.stderr)
        return 2
    clientes = pd.read_csv(ARQUIVO)
    colunas = [c for c in clientes.columns if c not in FORA]
    if len(args) != 2 or args[0] == args[1] or any(c not in colunas for c in args):
        print('erro: diga duas colunas diferentes, entre: ' + ', '.join(colunas), file=sys.stderr)
        return 2
    a, b = args
    if tirar:
        clientes = clientes.drop(clientes.nlargest(tirar, 'gasto_12m').index)
    n, c = matriz(clientes, a, b)
    print(f'clientes usados: {n}' + (f' (sem os {tirar} maiores gastos)' if tirar else ''))
    print(f'colunas: x = {a}, y = {b}, cada uma em escala z')
    print('matriz de covariância')
    for linha in c.values:
        print('  | ' + '  '.join(f'{v:7.4f}'.replace('.', ',') for v in linha) + ' |')
    print('para ver as direções: python3 ver_matriz.py ' + ' '.join(f'{v:.4f}' for v in c.values.ravel()))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
