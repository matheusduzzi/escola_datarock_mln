"""As três notas de estranheza num mapa de duas colunas, onde dá para ver quem é estranho.

    python3 formas.py                       # os três métodos com os parâmetros padrão
    python3 formas.py --vizinhos 5          # LOF com 5 vizinhos
    python3 formas.py --nu 0.2 --gamma 2    # One-Class SVM com outro nu e outro gamma
    python3 formas.py --pontos              # lista os pontos com as duas colunas

O mapa tem 252 pontos: uma mancha cheia (120), uma mancha espalhada (120), um grupinho
apertado longe das duas (8) e quatro pontos soltos com nome (P1 a P4). Nenhuma coluna diz quem é
estranho: os métodos só veem as duas coordenadas.
"""
import argparse
import sys

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

SOLTOS = {
    'P1': (1.3, 0.0),    # do lado da mancha cheia, fora dela; a essa distância, na espalhada, seria comum
    'P2': (2.5, 6.0),    # longe de tudo
    'P3': (8.6, 0.0),    # na beira da mancha espalhada, a 3 desvios do centro dela
    'P4': (2.6, 0.0),    # entre as duas manchas
}


def mapa():
    """Os 252 pontos, sempre os mesmos. Devolve coordenadas e o nome do grupo de cada ponto."""
    rng = np.random.default_rng(11)
    cheia = rng.normal((0.0, 0.0), 0.3, size=(120, 2))
    espalhada = rng.normal((5.0, 0.0), 1.2, size=(120, 2))
    grupinho = rng.normal((9.0, 5.0), 0.1, size=(8, 2))
    soltos = np.array(list(SOLTOS.values()))
    pontos = np.vstack([cheia, espalhada, grupinho, soltos])
    grupos = ['cheia'] * 120 + ['espalhada'] * 120 + ['grupinho'] * 8 + list(SOLTOS)
    return pontos, grupos


def numero(texto):
    return float(str(texto).replace(',', '.'))


def fmt(x, casas=2):
    return f'{x:.{casas}f}'.replace('.', ',')


def notas(pontos, vizinhos, nu, gamma, sorteio):
    """Nota de estranheza de cada método: maior quer dizer mais estranho (o sinal de score_samples trocado)."""
    floresta = IsolationForest(random_state=sorteio).fit(pontos)
    lof = LocalOutlierFactor(n_neighbors=vizinhos).fit(pontos)
    svm = OneClassSVM(nu=nu, gamma=gamma).fit(pontos)
    return {
        'Isolation Forest': -floresta.score_samples(pontos),
        'LOF': -lof.negative_outlier_factor_,
        'One-Class SVM': -svm.score_samples(pontos),
    }


def posicoes(nota):
    """Posição de cada ponto na ordem da nota (1 = o mais estranho). Empate: o ponto que vem antes."""
    ordem = np.lexsort((np.arange(len(nota)), -nota))
    pos = np.empty(len(nota), dtype=int)
    pos[ordem] = np.arange(1, len(nota) + 1)
    return pos


def main(argv=None):
    p = argparse.ArgumentParser(description='Isolation Forest, LOF e One-Class SVM num mapa de duas colunas.')
    p.add_argument('--vizinhos', default='20', help='vizinhos do LOF (n_neighbors), padrão 20')
    p.add_argument('--nu', default='0.05', help='nu do One-Class SVM, entre 0 e 1, padrão 0,05')
    p.add_argument('--gamma', default='scale', help="gamma do One-Class SVM: um número maior que 0 ou 'scale' (padrão)")
    p.add_argument('--sorteio', default='0', help='número do sorteio das árvores do Isolation Forest, padrão 0')
    p.add_argument('--pontos', action='store_true', help='lista os pontos com as duas colunas e sai')
    a = p.parse_args(argv)
    try:
        vizinhos = int(a.vizinhos)
        nu = numero(a.nu)
        gamma = a.gamma if a.gamma == 'scale' else numero(a.gamma)
        sorteio = int(a.sorteio)
    except ValueError:
        p.error('--vizinhos e --sorteio são números inteiros; --nu e --gamma são números (ou --gamma scale)')
    if not 1 <= vizinhos < 252:
        p.error('--vizinhos vai de 1 a 251')
    if not 0 < nu <= 1:
        p.error('--nu precisa ser maior que 0 e no máximo 1')
    if gamma != 'scale' and gamma <= 0:
        p.error('--gamma precisa ser maior que 0')

    pontos, grupos = mapa()
    if a.pontos:
        print('ponto  grupo        coluna 1  coluna 2')
        for i, (g, (x, y)) in enumerate(zip(grupos, pontos), start=1):
            print(f'{i:5d}  {g:<11}  {fmt(x):>8}  {fmt(y):>8}')
        return 0

    print('Mapa: 252 pontos (mancha cheia 120, mancha espalhada 120, grupinho 8, soltos P1 a P4)')
    print(f'parâmetros: LOF com {vizinhos} vizinhos; One-Class SVM com nu {fmt(nu)} e gamma '
          f'{gamma if gamma == "scale" else fmt(gamma)}; Isolation Forest sorteio {sorteio}')
    print()
    resultado = notas(pontos, vizinhos, nu, gamma, sorteio)
    g = np.array(grupos)
    print('posição na ordem da nota (1 = o mais estranho de 252)')
    print(f'{"":17}{"P1":>5}{"P2":>5}{"P3":>5}{"P4":>5}  grupinho (melhor a pior)  nas 10 primeiras')
    for metodo, nota in resultado.items():
        pos = posicoes(nota)
        soltos = ''.join(f'{pos[g == n][0]:5d}' for n in SOLTOS)
        gp = np.sort(pos[g == 'grupinho'])
        top = [grupos[i] for i in np.argsort(pos)[:10]]
        resumo = ', '.join(f'{n} {top.count(n)}' for n in ['cheia', 'espalhada', 'grupinho'] if top.count(n))
        soltos_top = [n for n in SOLTOS if n in top]
        if soltos_top:
            resumo += ', ' + ' '.join(soltos_top)
        print(f'{metodo:<17}{soltos}  {gp[0]:>3} a {gp[-1]:<3}{"":15}  {resumo}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
