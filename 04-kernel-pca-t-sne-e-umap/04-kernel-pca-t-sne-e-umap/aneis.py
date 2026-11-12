"""Dois anéis, um dentro do outro: o que o PCA e o Kernel PCA fazem com eles.

    python3 aneis.py

Só usa o NumPy. Os pontos saem iguais toda vez.
"""
import numpy as np

PONTOS_POR_ANEL = 60
RAIOS = (1.0, 3.0)
GAMMAS = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)


def br(x, casas=2):
    return f'{x:.{casas}f}'.replace('.', ',')


def milhar(n):
    return f'{n:,}'.replace(',', '.')


def titulo(texto):
    print(texto)
    print('-' * len(texto))


def pontos():
    rng = np.random.default_rng(4)
    angulos = np.linspace(0, 2 * np.pi, PONTOS_POR_ANEL, endpoint=False)
    aneis = [np.c_[r * np.cos(angulos), r * np.sin(angulos)] + rng.normal(0, 0.15, (PONTOS_POR_ANEL, 2))
             for r in RAIOS]
    rotulo = np.r_[np.zeros(PONTOS_POR_ANEL), np.ones(PONTOS_POR_ANEL)]
    return np.r_[aneis[0], aneis[1]], rotulo


def distancias2(a):
    s = (a * a).sum(axis=1)
    d = s[:, None] + s[None, :] - 2 * np.dot(a, a.T)
    np.maximum(d, 0, out=d)
    np.fill_diagonal(d, 0)
    return d


def lado_errado(valores, rotulo):
    """Menor número de pontos do lado errado, no melhor corte de um eixo só."""
    ordem = rotulo[np.argsort(valores)]
    n = len(ordem)
    melhor = n
    for k in range(n + 1):
        erros = int((ordem[:k] == 1).sum() + (ordem[k:] == 0).sum())
        melhor = min(melhor, erros, n - erros)
    return melhor


def kernel_pca(x, gamma, k=3):
    semelhanca = np.exp(-gamma * distancias2(x))
    n = len(semelhanca)
    um = np.ones((n, n)) / n
    centrada = semelhanca - np.dot(um, semelhanca) - np.dot(semelhanca, um) + np.dot(np.dot(um, semelhanca), um)
    valores, vetores = np.linalg.eigh(centrada)
    valores, vetores = valores[::-1], vetores[:, ::-1]
    return vetores[:, :k] * np.sqrt(np.maximum(valores[:k], 0))


def main():
    x, rotulo = pontos()
    n = len(x)
    titulo('1. Os pontos')
    for r, nome in zip(RAIOS, ('de dentro', 'de fora')):
        print(f'  anel {nome}: {PONTOS_POR_ANEL} pontos, raio {br(r, 0)}')
    print()

    titulo('2. PCA: o melhor eixo reto')
    centrado = x - x.mean(axis=0)
    _, s, vt = np.linalg.svd(centrado, full_matrices=False)
    variancia = s ** 2 / (n - 1)
    for c in range(2):
        nota = np.dot(centrado, vt[c])
        print(f'  componente {c + 1}: {br(100 * variancia[c] / variancia.sum(), 1)}% da variância, '
              f'{lado_errado(nota, rotulo)} pontos do lado errado do melhor corte')
    print()

    titulo('3. Kernel PCA com o kernel RBF, um gamma de cada vez')
    print('  pontos do lado errado do melhor corte em cada componente (0 = um corte separa os anéis)')
    print('  gamma   comp 1  comp 2  comp 3  | semelhança de dois pontos a distância 1 e a distância 2')
    for g in GAMMAS:
        notas = kernel_pca(x, g)
        erros = [lado_errado(notas[:, c], rotulo) for c in range(3)]
        print(f'  {br(g):>5}   {erros[0]:>6}  {erros[1]:>6}  {erros[2]:>6}  |  '
              f'{br(np.exp(-g * 1.0), 3)} e {br(np.exp(-g * 4.0), 3)}')
    print()

    titulo('4. O tamanho da tabela de semelhanças (uma linha e uma coluna por ponto)')
    for linhas, nome in ((n, 'pontos dos anéis'), (3000, 'clientes da loja'), (100_000, 'clientes')):
        numeros = linhas * linhas
        bytes_ = 8 * numeros
        unidade = 'KB' if bytes_ < 1e6 else ('MB' if bytes_ < 1e9 else 'GB')
        tamanho = bytes_ / {'KB': 1e3, 'MB': 1e6, 'GB': 1e9}[unidade]
        print(f'  {milhar(linhas)} {nome}: {milhar(numeros)} números, {br(tamanho, 0)} {unidade}')


if __name__ == '__main__':
    main()
