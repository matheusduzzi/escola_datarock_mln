"""O PCA de quatro clientes, passo a passo, com as mesmas contas que você faz à mão.

    python3 quatro_clientes.py

Duas colunas: pedidos e categorias nos 12 meses. Os quatro clientes estão no próprio programa.
"""
import numpy as np

NOMES = ['A', 'B', 'C', 'D']
TABELA = np.array([[4.0, 2.0],
                   [5.0, 1.0],
                   [7.0, 5.0],
                   [8.0, 4.0]])


def br(x, casas=2):
    return f'{x:.{casas}f}'.replace('.', ',')


def ponto(v, casas=2):
    return '(' + '; '.join(br(x, casas) for x in v) + ')'


def com_sinal(v):
    """Põe positivo o maior número da direção. O sinal é uma escolha: a reta é a mesma."""
    return v if v[np.argmax(np.abs(v))] > 0 else -v


def titulo(texto):
    print()
    print(texto)
    print('-' * len(texto))


def main():
    n = len(TABELA)
    titulo('1. Os quatro clientes (x = pedidos, y = categorias)')
    for nome, linha in zip(NOMES, TABELA):
        print(f'  {nome} {ponto(linha, 0)}')

    titulo('2. Centralizar: tirar a média de cada coluna')
    media = TABELA.mean(axis=0)
    centro = TABELA - media
    print(f'  média {ponto(media)}')
    for nome, linha in zip(NOMES, centro):
        print(f'  {nome} {ponto(linha)}')

    titulo('3. Matriz de covariância (divide por n - 1 = 3)')
    cov = np.dot(centro.T, centro) / (n - 1)
    for linha in cov:
        print('  | ' + '  '.join(br(x) for x in linha) + ' |')

    titulo('4. Direções e variância de cada uma (autovetores e autovalores)')
    valores, vetores = np.linalg.eigh(cov)
    ordem = np.argsort(valores)[::-1]
    valores, vetores = valores[ordem], vetores[:, ordem]
    total = valores.sum()
    for i in range(2):
        d = com_sinal(vetores[:, i])
        vetores[:, i] = d
        print(f'  componente {i + 1}: direção {ponto(d)}, variância {br(valores[i])}, '
              f'{br(100 * valores[i] / total, 1)}% do total {br(total)}')

    titulo('5. Guardar só o componente 1 e reconstruir')
    d1 = vetores[:, 0]
    notas = np.dot(centro, d1)
    volta = media + np.outer(notas, d1)
    erros = ((TABELA - volta) ** 2).sum(axis=1)
    for nome, original, nota, rec, erro in zip(NOMES, TABELA, notas, volta, erros):
        print(f'  {nome} {ponto(original, 0)}: nota {br(nota)}, volta como {ponto(rec)}, erro {br(erro)}')
    print(f'  erro somado {br(erros.sum())} = variância do componente 2 x (n - 1) = '
          f'{br(valores[1])} x {n - 1} = {br(valores[1] * (n - 1))}')

    titulo('6. O mesmo sem centralizar')
    _, s, vt = np.linalg.svd(TABELA, full_matrices=False)
    d = com_sinal(vt[0])
    fracao = s[0] ** 2 / (s ** 2).sum()
    print(f'  direção do componente 1: {ponto(d)}, com {br(100 * fracao, 1)}% da soma dos quadrados')
    print(f'  direção da média:        {ponto(media / np.linalg.norm(media))}')
    print(f'  direção certa, centralizada: {ponto(d1)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
