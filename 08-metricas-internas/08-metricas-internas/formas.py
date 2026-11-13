"""As três notas em formas de grupo que dá para desenhar.

    python3 formas.py                  # as duas formas
    python3 formas.py --forma aneis    # só os anéis

As formas:
- aneis: os dois anéis das aulas «Kernel PCA, t-SNE e UMAP» e «Densidade e hierarquia», 60 pontos cada,
  raios 1 e 3; o grupo 1 é o anel de dentro;
- manchas: três manchas redondas de 60 pontos, bem separadas.

Aqui a gente sabe de que grupo cada ponto veio, porque as formas foram montadas assim. Isso serve
para ver o que as notas premiam. Nos clientes, ninguém sabe.
Os pontos saem iguais toda vez. Rode de dentro da pasta da aula.
"""
import argparse
import sys
import warnings

import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_samples, silhouette_score

NOMES = {'aneis': 'Dois anéis', 'manchas': 'Três manchas redondas'}
RAIO = {'aneis': 1.0, 'manchas': 1.0}
MINIMO = 5


def br(x, casas=1):
    texto = f'{float(x):.{casas}f}'.replace('.', ',')
    return texto[1:] if texto.startswith('-') and set(texto[1:]) <= set('0,') else texto


def aneis():
    rng = np.random.default_rng(4)
    angulos = np.linspace(0, 2 * np.pi, 60, endpoint=False)
    partes = [np.c_[r * np.cos(angulos), r * np.sin(angulos)] + rng.normal(0, 0.15, (60, 2)) for r in (1.0, 3.0)]
    return np.r_[partes[0], partes[1]], np.r_[np.zeros(60), np.ones(60)].astype(int)


def manchas():
    rng = np.random.default_rng(11)
    centros = [(0, 0), (5, 0), (2.5, 4.5)]
    partes = [rng.normal(c, 0.6, (60, 2)) for c in centros]
    return np.r_[tuple(partes)], np.repeat(np.arange(3), 60)


GERAR = {'aneis': aneis, 'manchas': manchas}


def errados(rotulos, verdade):
    """Pontos fora do grupo da maioria da sua origem; ruído conta como errado."""
    total = 0
    for g in set(rotulos):
        nele = rotulos == g
        total += int(nele.sum()) if g == -1 else int(nele.sum() - np.bincount(verdade[nele]).max())
    return total


def mostrar(forma):
    x, verdade = GERAR[forma]()
    k = len(set(verdade))
    linhas = [('a origem dos pontos', verdade),
              (f'K-means, K = {k}', KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(x)),
              (f'DBSCAN, raio {br(RAIO[forma])}, mín. {MINIMO}', DBSCAN(eps=RAIO[forma], min_samples=MINIMO).fit_predict(x))]
    titulo = f'{NOMES[forma]}: {len(x)} pontos, {k} grupos de verdade'
    print(titulo)
    print('-' * len(titulo))
    print('grupos feitos por            grupos  errados  silhouette  Davies-Bouldin  Calinski-Harabasz')
    for nome, rot in linhas:
        rot = np.asarray(rot)
        if (rot == -1).any():
            raise SystemExit('erro: o DBSCAN deixou pontos no ruído; as notas abaixo supõem todos os pontos em grupo')
        s, d, c = silhouette_score(x, rot), davies_bouldin_score(x, rot), calinski_harabasz_score(x, rot)
        print(f'{nome:<28} {len(set(rot)):>6} {errados(rot, verdade):>8} {br(s, 3):>11} {br(d, 3):>15} {br(c, 1):>18}')
    s = silhouette_samples(x, verdade)
    partes = [f'grupo {g + 1}, {br(s[verdade == g].mean(), 3)}, com {int((s[verdade == g] < 0).sum())} de '
              f'{int((verdade == g).sum())} pontos abaixo de 0' for g in sorted(set(verdade))]
    print('na origem dos pontos, a silhouette de cada grupo: ' + '; '.join(partes))
    print()


def main(argv):
    p = argparse.ArgumentParser(description='As três notas em formas que dá para desenhar.')
    p.add_argument('--forma', choices=list(GERAR) + ['todas'], default='todas')
    a = p.parse_args(argv)
    for forma in (GERAR if a.forma == 'todas' else [a.forma]):
        mostrar(forma)
    return 0


if __name__ == '__main__':
    warnings.simplefilter('error')
    sys.exit(main(sys.argv[1:]))
