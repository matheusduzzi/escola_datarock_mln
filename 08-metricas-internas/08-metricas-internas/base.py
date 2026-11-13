"""Silhouette, Davies-Bouldin e Calinski-Harabasz nos 3000 clientes da Loja Exemplo.

    python3 base.py --propostas                          # as três propostas da aula, cada uma medida onde foi feita
    python3 base.py --propostas --medido-em z            # as mesmas três, medidas todas na escala z
    python3 base.py --grupos ward --k 5 --por-grupo      # a silhouette de cada grupo e quantos clientes ficam abaixo de 0
    python3 base.py --grupos kmeans --feito-em reais --k 3 --medido-em todas
    python3 base.py --grupos dbscan --medido-em z        # DBSCAN com raio 2,5: o ruído como grupo e o ruído fora
    python3 base.py --curva ward                         # as três notas para 2 a 10 grupos

Onde os grupos são feitos (--feito-em) e onde as notas são medidas (--medido-em):
- z: as oito colunas de comportamento na escala z, como na aula «PCA»;
- pca2: os dois primeiros componentes do PCA dessas oito colunas na escala z;
- reais: as oito colunas como estão no arquivo, em reais, dias, pedidos e porcentagens.
Rode de dentro da pasta da aula.
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_samples, silhouette_score

ARQUIVO = Path('..') / 'dados' / 'clientes' / 'clientes.csv'
COLUNAS = ['pedidos_12m', 'gasto_12m', 'ticket_medio', 'dias_desde_ultima_compra',
           'categorias_distintas', 'pct_pedidos_cupom', 'pct_pedidos_app', 'devolucoes_12m']
CURTOS = ['pedidos', 'gasto R$', 'ticket R$', 'dias', 'categ.', 'cupom %', 'app %', 'devol.']
ESPACOS = {'z': 'oito colunas na escala z', 'pca2': 'dois componentes do PCA', 'reais': 'oito colunas em reais'}
TOPO = 40
PROPOSTAS = [('agência', 'kmeans', 'reais', 3), ('BI', 'kmeans', 'pca2', 3), ('Ward', 'ward', 'z', 5)]


class Erro(Exception):
    pass


def br(x, casas=1):
    texto = f'{float(x):.{casas}f}'.replace('.', ',')
    return texto[1:] if texto.startswith('-') and set(texto[1:]) <= set('0,') else texto


def milhar(n):
    return f'{int(round(n)):,}'.replace(',', '.')


def numero(texto):
    try:
        return float(texto.replace(',', '.'))
    except ValueError:
        raise argparse.ArgumentTypeError(f'"{texto}" não é um número')


def ler_argumentos(argv):
    p = argparse.ArgumentParser(description='Métricas internas nos clientes da Loja Exemplo.')
    modo = p.add_mutually_exclusive_group(required=True)
    modo.add_argument('--propostas', action='store_true', help='as três propostas da aula')
    modo.add_argument('--grupos', choices=['kmeans', 'ward', 'dbscan'], help='o método que faz os grupos')
    modo.add_argument('--curva', choices=['kmeans', 'ward'], help='as três notas para 2 a 10 grupos')
    p.add_argument('--feito-em', choices=list(ESPACOS), help='onde os grupos são feitos (padrão: z)')
    p.add_argument('--medido-em', choices=list(ESPACOS) + ['todas'],
                   help='onde as notas são medidas (padrão: onde os grupos foram feitos)')
    p.add_argument('--k', type=int, help='quantos grupos, de 2 a 10 (kmeans e ward; padrão 5)')
    p.add_argument('--raio', type=numero, help='raio do DBSCAN (padrão 2,5)')
    p.add_argument('--min', type=int, help='mínimo de clientes no raio do DBSCAN (padrão 8)')
    p.add_argument('--por-grupo', action='store_true', help='a silhouette de cada grupo e os clientes abaixo de 0')
    p.add_argument('--sem-topo', action='store_true', help=f'tira os clientes de {TOPO} ou mais pedidos')
    p.add_argument('--salvar', metavar='ARQUIVO.csv', help='grava id_cliente, grupo e a silhouette de cada cliente')
    return p.parse_args(argv)


def validar(a):
    if a.k is not None and a.grupos not in ('kmeans', 'ward'):
        raise Erro('--k só vale com --grupos kmeans ou --grupos ward')
    if a.k is not None and not 2 <= a.k <= 10:
        raise Erro('--k precisa ficar entre 2 e 10')
    if (a.raio is not None or a.min is not None) and a.grupos != 'dbscan':
        raise Erro('--raio e --min só valem com --grupos dbscan')
    if a.raio is not None and not 0 < a.raio <= 10:
        raise Erro('o --raio precisa ser maior que 0 e no máximo 10')
    if a.min is not None and not 2 <= a.min <= 100:
        raise Erro('--min precisa ficar entre 2 e 100')
    if a.feito_em is not None and a.propostas:
        raise Erro('--feito-em não vale com --propostas: cada proposta já diz onde foi feita')
    if a.feito_em is not None and a.curva:
        raise Erro('--feito-em não vale com --curva: a curva faz os grupos na escala z')
    if a.medido_em == 'todas' and (a.propostas or a.curva):
        raise Erro('--medido-em todas só vale com --grupos')
    if a.por_grupo and not a.grupos:
        raise Erro('--por-grupo só vale com --grupos')
    if a.por_grupo and a.medido_em == 'todas':
        raise Erro('--por-grupo precisa de uma representação só em --medido-em')
    if a.salvar and not a.grupos:
        raise Erro('--salvar só vale com --grupos')
    if a.salvar and a.medido_em == 'todas':
        raise Erro('--salvar precisa de uma representação só em --medido-em')
    if a.salvar and not a.salvar.endswith('.csv'):
        raise Erro('--salvar precisa de um nome que termine em .csv')


def ler_clientes():
    if not ARQUIVO.is_file():
        raise Erro(f'não achei {ARQUIVO}. Rode de dentro da pasta da aula, 08-metricas-internas.')
    return pd.read_csv(ARQUIVO)


def espacos(base):
    reais = base[COLUNAS].to_numpy(float)
    z = (reais - reais.mean(axis=0)) / reais.std(axis=0, ddof=1)
    pca2 = PCA(n_components=2).fit_transform(z)
    return {'z': z, 'pca2': pca2, 'reais': reais}


def agrupar(metodo, x, k=5, raio=2.5, minimo=8):
    if metodo == 'kmeans':
        return KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(x)
    if metodo == 'ward':
        return fcluster(linkage(x, 'ward'), k, 'maxclust') - 1
    return DBSCAN(eps=raio, min_samples=minimo).fit_predict(x)


def em_ordem(rotulos):
    """Renumera os grupos do maior para o menor, a partir de 1; o ruído continua -1."""
    tamanhos = pd.Series(rotulos[rotulos >= 0]).value_counts()
    ordem = sorted(tamanhos.index, key=lambda g: (-tamanhos[g], g))
    mapa = {g: n + 1 for n, g in enumerate(ordem)}
    return np.array([mapa.get(r, -1) for r in rotulos])


def notas(x, rotulos):
    return (silhouette_score(x, rotulos), davies_bouldin_score(x, rotulos), calinski_harabasz_score(x, rotulos))


def texto_notas(n):
    return f'{br(n[0], 3):>10} {br(n[1], 3):>14} {milhar(n[2]):>17}'


def titulo(texto):
    print(texto)
    print('-' * len(texto))


def tamanhos(rotulos):
    t = pd.Series(rotulos[rotulos >= 0]).value_counts().sort_index()
    return ', '.join(str(v) for v in t.to_numpy())


CABECALHO = 'silhouette  Davies-Bouldin  Calinski-Harabasz'


def mostrar_propostas(base, esp, medido_em):
    if medido_em:
        titulo(f'As três propostas, medidas todas em: {ESPACOS[medido_em]}')
    else:
        titulo('As três propostas, cada uma medida onde foi feita')
    print(f'proposta   método e onde foi feita                    tamanhos dos grupos     medida em  {CABECALHO}')
    for nome, metodo, feito, k in PROPOSTAS:
        rot = em_ordem(agrupar(metodo, esp[feito], k))
        onde = medido_em or feito
        descricao = f'{"K-means" if metodo == "kmeans" else "Ward"} K = {k}, {ESPACOS[feito]}'
        print(f'{nome:<10} {descricao:<42} {tamanhos(rot):<23} {onde:>9}  {texto_notas(notas(esp[onde], rot))}')
    print()
    print('Na silhouette, maior é melhor (de -1 a 1). No Davies-Bouldin, menor é melhor (0 é o mínimo).')
    print('No Calinski-Harabasz, maior é melhor, sem teto.')


def tabela_grupos(base, rotulos, s):
    print('grupo  clientes  do topo  silhouette  abaixo de 0  ' + ''.join(f'{c:>10}' for c in CURTOS) + '   <- média do grupo')
    for g in sorted(set(rotulos) - {-1}) + ([-1] if (rotulos == -1).any() else []):
        nele = rotulos == g
        medias = base.loc[nele, COLUNAS].mean()
        valores = [br(medias.iloc[0]), milhar(medias.iloc[1]), milhar(medias.iloc[2]), milhar(medias.iloc[3]),
                   br(medias.iloc[4]), milhar(medias.iloc[5]), milhar(medias.iloc[6]), br(medias.iloc[7])]
        nome = 'ruído' if g == -1 else str(g)
        sil = '-' if g == -1 else br(s[nele].mean(), 3)
        abaixo = '-' if g == -1 else str(int((s[nele] < 0).sum()))
        topo = int((base.loc[nele, 'pedidos_12m'] >= TOPO).sum())
        print(f'{nome:>5} {int(nele.sum()):>9} {topo:>8} {sil:>11} {abaixo:>12}  ' + ''.join(f'{v:>10}' for v in valores))


def mostrar_grupos(base, esp, a):
    feito = a.feito_em or 'z'
    k = a.k or 5
    if a.grupos == 'dbscan':
        raio, minimo = (2.5 if a.raio is None else a.raio), (8 if a.min is None else a.min)
        rot = em_ordem(agrupar('dbscan', esp[feito], raio=raio, minimo=minimo))
        descricao = f'DBSCAN com raio {br(raio, 2)} e mínimo {minimo}, {ESPACOS[feito]}'
    else:
        rot = em_ordem(agrupar(a.grupos, esp[feito], k))
        descricao = f'{"K-means" if a.grupos == "kmeans" else "Ward"} K = {k}, {ESPACOS[feito]}'
    ruido = int((rot == -1).sum())
    grupos = len(set(rot) - {-1})
    if grupos + (1 if ruido else 0) < 2:
        raise Erro(f'esse DBSCAN deu {grupos} grupo(s) e {ruido} cliente(s) no ruído; as notas precisam de pelo '
                   'menos 2 grupos. Mude o raio ou o mínimo')
    titulo(descricao)
    print(f'tamanhos dos grupos: {tamanhos(rot) or "nenhum grupo"}' + (f'; ruído: {ruido} cliente(s)' if ruido else ''))
    onde = [a.medido_em] if a.medido_em and a.medido_em != 'todas' else (list(ESPACOS) if a.medido_em == 'todas' else [feito])
    print()
    print(f'medida em  {"o ruído":<22} {CABECALHO}')
    for e in onde:
        if ruido:
            print(f'{e:>9}  {"entra como um grupo":<22} {texto_notas(notas(esp[e], rot))}')
            dentro = rot >= 0
            if grupos >= 2:
                print(f'{e:>9}  {"fica fora da conta":<22} {texto_notas(notas(esp[e][dentro], rot[dentro]))}')
            else:
                print(f'{e:>9}  {"fica fora da conta":<22} {"sem nota: sobra um grupo só":>43}')
        else:
            print(f'{e:>9}  {"(sem ruído)":<22} {texto_notas(notas(esp[e], rot))}')
    if a.por_grupo or a.salvar:
        e = onde[0]
        s = silhouette_samples(esp[e], rot)
        if a.por_grupo:
            print()
            titulo(f'Cada grupo, com a silhouette medida em: {ESPACOS[e]}')
            if ruido:
                print('o ruído entra na conta como um grupo; a linha dele não tem silhouette')
            print(f'a coluna "do topo" conta os clientes de {TOPO} ou mais pedidos no ano')
            tabela_grupos(base, rot, s)
        if a.salvar:
            pd.DataFrame({'id_cliente': base['id_cliente'], 'grupo': rot, 'silhouette': np.round(s, 4)}).to_csv(
                a.salvar, index=False)
            print(f'\ngravei {len(base)} linhas em {a.salvar}')


def mostrar_curva(esp, metodo, medido_em):
    onde = medido_em or 'z'
    titulo(f'{"K-means" if metodo == "kmeans" else "Ward"} feito nas {ESPACOS["z"]}, notas medidas em: {ESPACOS[onde]}')
    print(f' K  menor grupo  {CABECALHO}')
    linhas = []
    for k in range(2, 11):
        rot = agrupar(metodo, esp['z'], k)
        n = notas(esp[onde], rot)
        linhas.append((k, n))
        print(f'{k:>2} {int(np.bincount(rot[rot >= 0]).min()):>12}  {texto_notas(n)}')
    melhor_s = max(linhas, key=lambda l: l[1][0])[0]
    melhor_d = min(linhas, key=lambda l: l[1][1])[0]
    melhor_c = max(linhas, key=lambda l: l[1][2])[0]
    print(f'melhor K por nota: silhouette K = {melhor_s}; Davies-Bouldin K = {melhor_d}; Calinski-Harabasz K = {melhor_c}')


def main(argv):
    a = ler_argumentos(argv)
    try:
        validar(a)
        base = ler_clientes()
        if a.sem_topo:
            base = base[base['pedidos_12m'] < TOPO].reset_index(drop=True)
        esp = espacos(base)
        if a.propostas:
            mostrar_propostas(base, esp, a.medido_em)
        elif a.curva:
            mostrar_curva(esp, a.curva, a.medido_em)
        else:
            mostrar_grupos(base, esp, a)
    except Erro as e:
        print(f'erro: {e}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    warnings.simplefilter('error')
    sys.exit(main(sys.argv[1:]))
