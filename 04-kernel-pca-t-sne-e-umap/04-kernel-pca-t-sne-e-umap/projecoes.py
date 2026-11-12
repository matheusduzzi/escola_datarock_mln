"""PCA, Kernel PCA e t-SNE nos clientes da Loja Exemplo, só com o NumPy e o pandas.

    python3 projecoes.py                              # PCA nos 1.000 clientes do sorteio
    python3 projecoes.py --comparar                   # os três métodos lado a lado
    python3 projecoes.py --metodo kpca --gamma 0,01   # Kernel PCA com outro gamma
    python3 projecoes.py --metodo tsne --perplexidade 5
    python3 projecoes.py --metodo tsne --repetir      # dois inícios sorteados, o mesmo desenho?
    python3 projecoes.py --metodo tsne --mais 50      # 50 clientes novos chegam
    python3 projecoes.py --metodo tsne --salvar tsne.csv

As oito colunas de comportamento entram na escala z, como na aula «PCA».
Rode de dentro da pasta da aula.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ARQUIVO = Path('..') / 'dados' / 'clientes' / 'clientes.csv'
COLUNAS = ['pedidos_12m', 'gasto_12m', 'ticket_medio', 'dias_desde_ultima_compra',
           'categorias_distintas', 'pct_pedidos_cupom', 'pct_pedidos_app', 'devolucoes_12m']
NOMES = {'pca': 'PCA', 'kpca': 'Kernel PCA (RBF)', 'tsne': 't-SNE'}
CLIENTES = 1000
VIZINHOS = 10
GRANDES = 40          # clientes com 40 ou mais pedidos no ano
PASSOS = 1000         # passos do t-SNE, como o max_iter do scikit-learn
EXAGERO, PASSOS_EXAGERO = 12.0, 250


class Erro(Exception):
    pass


def br(x, casas=2):
    return f'{float(x):.{casas}f}'.replace('.', ',').replace('-0,' + '0' * casas, '0,' + '0' * casas)


def numero(texto):
    try:
        return float(texto.replace(',', '.'))
    except ValueError:
        raise argparse.ArgumentTypeError(f'"{texto}" não é número')


def ler_argumentos(argv):
    p = argparse.ArgumentParser(description='PCA, Kernel PCA e t-SNE nos clientes da Loja Exemplo.')
    p.add_argument('--metodo', choices=['pca', 'kpca', 'tsne'], default='pca')
    p.add_argument('--comparar', action='store_true', help='roda PCA, Kernel PCA e t-SNE com os padrões')
    p.add_argument('--gamma', type=numero, help='gamma do Kernel PCA (padrão: 1 dividido pelo número de colunas)')
    p.add_argument('--perplexidade', type=numero, help='perplexidade do t-SNE (padrão: 30)')
    p.add_argument('--inicio', choices=['pca', 'aleatorio'], default='pca', help='como o t-SNE começa')
    p.add_argument('--sorteio', type=int, default=1, help='sorteio do início aleatório do t-SNE')
    p.add_argument('--repetir', action='store_true', help='t-SNE duas vezes, com os inícios sorteados 1 e 2')
    p.add_argument('--mais', type=int, default=0, metavar='N', help='N clientes novos chegam depois do desenho')
    p.add_argument('--clientes', type=int, default=CLIENTES, help='quantos clientes sortear (de 100 a 3000)')
    p.add_argument('--salvar', metavar='ARQUIVO.csv', help='grava id_cliente, x e y do desenho')
    return p.parse_args(argv)


def validar(a):
    if not 100 <= a.clientes <= 3000:
        raise Erro('--clientes precisa ficar entre 100 e 3000')
    if a.comparar and (a.metodo != 'pca' or a.gamma is not None or a.perplexidade is not None
                       or a.repetir or a.mais or a.salvar or a.inicio != 'pca'):
        raise Erro('--comparar roda os três métodos com os padrões; use sem outras opções (só --clientes)')
    if a.gamma is not None and a.metodo != 'kpca':
        raise Erro('--gamma só vale com --metodo kpca')
    if a.gamma is not None and a.gamma <= 0:
        raise Erro('--gamma precisa ser maior que zero')
    if (a.perplexidade is not None or a.repetir or a.inicio != 'pca') and a.metodo != 'tsne':
        raise Erro('--perplexidade, --inicio e --repetir só valem com --metodo tsne')
    if a.perplexidade is not None and not 2 <= a.perplexidade < a.clientes / 3:
        raise Erro(f'--perplexidade precisa ficar entre 2 e {br(a.clientes / 3, 1)} '
                   f'(menos que um terço dos {a.clientes} clientes)')
    if a.repetir and (a.mais or a.salvar):
        raise Erro('--repetir não se combina com --mais nem com --salvar')
    if a.mais < 0 or a.mais > 3000 - a.clientes:
        raise Erro(f'--mais precisa ficar entre 1 e {3000 - a.clientes}, os clientes fora do sorteio')
    if a.salvar and not a.salvar.endswith('.csv'):
        raise Erro('--salvar precisa de um nome terminado em .csv')


def ler_clientes():
    if not ARQUIVO.is_file():
        raise Erro(f'não achei {ARQUIVO}. Rode de dentro da pasta da aula, 04-kernel-pca-t-sne-e-umap.')
    return pd.read_csv(ARQUIVO)


def distancias2(a, b=None):
    b = a if b is None else b
    d = (a * a).sum(axis=1)[:, None] + (b * b).sum(axis=1)[None, :] - 2 * np.dot(a, b.T)
    np.maximum(d, 0, out=d)
    if b is a:
        np.fill_diagonal(d, 0)
    return d


# ---------- os três métodos ----------

def pca(z, novos=None):
    media = z.mean(axis=0)
    _, _, vt = np.linalg.svd(z - media, full_matrices=False)
    vt = vt[:2] * np.sign(vt[:2, np.abs(vt[:2]).argmax(axis=1)].diagonal())[:, None]
    desenho = np.dot(z - media, vt.T)
    return desenho, (None if novos is None else np.dot(novos - media, vt.T))


def kernel_pca(z, gamma, novos=None):
    semelhanca = np.exp(-gamma * distancias2(z))
    n = len(z)
    media_linha = semelhanca.mean(axis=0)
    media_total = semelhanca.mean()
    centrada = semelhanca - media_linha[None, :] - media_linha[:, None] + media_total
    valores, vetores = np.linalg.eigh(centrada)
    valores, vetores = valores[::-1][:2], vetores[:, ::-1][:, :2]
    vetores = vetores * np.sign(vetores[np.abs(vetores).argmax(axis=0), [0, 1]])[None, :]
    desenho = vetores * np.sqrt(valores)
    if novos is None:
        return desenho, None
    # cliente novo: a semelhança dele com cada um dos n clientes do desenho, centrada do mesmo jeito
    k = np.exp(-gamma * distancias2(novos, z))
    k = k - k.mean(axis=1)[:, None] - media_linha[None, :] + media_total
    return desenho, np.dot(k, vetores / np.sqrt(valores))


def afinidades(d2, perplexidade):
    n = len(d2)
    alvo = np.log(perplexidade)
    p = np.zeros((n, n))
    for i in range(n):
        fora = np.arange(n) != i
        d = d2[i, fora] - d2[i, fora].min()
        beta, baixo, alto = 1.0, 0.0, np.inf
        for _ in range(100):
            linha = np.exp(-d * beta)
            soma = linha.sum()
            entropia = np.log(soma) + beta * np.dot(d, linha) / soma
            if abs(entropia - alvo) < 1e-5:
                break
            if entropia > alvo:
                baixo = beta
                beta = beta * 2 if alto == np.inf else (beta + alto) / 2
            else:
                alto = beta
                beta = (beta + baixo) / 2
        p[i, fora] = linha / soma
    return np.maximum((p + p.T) / (2 * n), 1e-12)


def tsne(z, perplexidade, inicio='pca', sorteio=1):
    n = len(z)
    p = afinidades(distancias2(z), perplexidade)
    if inicio == 'pca':
        y = pca(z)[0]
        y = y / y[:, 0].std() * 1e-4
    else:
        y = np.random.default_rng(sorteio).normal(0, 1e-4, (n, 2))
    passo = max(n / EXAGERO / 4, 50)
    velocidade = np.zeros_like(y)
    ganho = np.ones_like(y)
    for t in range(PASSOS):
        exagero, impulso = (EXAGERO, 0.5) if t < PASSOS_EXAGERO else (1.0, 0.8)
        perto = 1 / (1 + distancias2(y))
        np.fill_diagonal(perto, 0)
        q = np.maximum(perto / perto.sum(), 1e-12)
        w = (exagero * p - q) * perto
        gradiente = 4 * (w.sum(axis=1)[:, None] * y - np.dot(w, y))
        mesmo = np.sign(gradiente) == np.sign(velocidade)
        ganho = np.maximum(np.where(mesmo, ganho * 0.8, ganho + 0.2), 0.01)
        velocidade = impulso * velocidade - passo * ganho * gradiente
        y = y + velocidade
    return y - y.mean(axis=0)


# ---------- as medidas ----------

def mais_proximos(d2, k=VIZINHOS):
    d = d2.copy()
    np.fill_diagonal(d, np.inf)
    return np.argpartition(d, k, axis=1)[:, :k]


def vizinhos_em_comum(d2_a, d2_b):
    a, b = mais_proximos(d2_a), mais_proximos(d2_b)
    return np.mean([len(set(x) & set(y)) / VIZINHOS for x, y in zip(a, b)])


def correlacao_distancias(d2_a, d2_b):
    i = np.triu_indices(len(d2_a), 1)
    return np.corrcoef(np.sqrt(d2_a[i]), np.sqrt(d2_b[i]))[0, 1]


def afastamento(pontos, grandes):
    """Distância do centro dos grandes até o centro dos outros, em distâncias típicas dos outros."""
    centro = pontos[~grandes].mean(axis=0)
    tipico = np.median(np.linalg.norm(pontos[~grandes] - centro, axis=1))
    return np.linalg.norm(pontos[grandes].mean(axis=0) - centro) / tipico


def medidas(z, desenho, grandes):
    dz, dd = distancias2(z), distancias2(desenho)
    return vizinhos_em_comum(dz, dd), correlacao_distancias(dz, dd), afastamento(desenho, grandes)


def projetar(metodo, z, a, novos=None):
    if metodo == 'pca':
        return pca(z, novos)
    if metodo == 'kpca':
        return kernel_pca(z, a.gamma if a.gamma is not None else 1 / z.shape[1], novos)
    return tsne(z, a.perplexidade if a.perplexidade is not None else 30.0, a.inicio, a.sorteio), None


def descricao(metodo, a, colunas):
    if metodo == 'pca':
        return 'PCA, 2 componentes'
    if metodo == 'kpca':
        g = a.gamma if a.gamma is not None else 1 / colunas
        return f'Kernel PCA (RBF), gamma {br(g, 4).rstrip("0").rstrip(",")}, 2 componentes'
    perp = a.perplexidade if a.perplexidade is not None else 30.0
    inicio = 'pelo PCA' if a.inicio == 'pca' else f'sorteado (sorteio {a.sorteio})'
    return f't-SNE, perplexidade {br(perp, 1).replace(",0", "")}, início {inicio}, {PASSOS} passos'


def main(argv):
    a = ler_argumentos(argv)
    validar(a)
    clientes = ler_clientes()
    x = clientes[COLUNAS].to_numpy(float)
    z_todos = (x - x.mean(axis=0)) / x.std(axis=0, ddof=1)
    sorteados = np.sort(np.random.default_rng(2026).choice(len(x), a.clientes, replace=False))
    z = z_todos[sorteados]
    grandes = x[sorteados, 0] >= GRANDES

    print(f'clientes: {a.clientes} sorteados dos {len(x)} (o mesmo sorteio toda vez), '
          f'{len(COLUNAS)} colunas na escala z')
    print(f'clientes de {GRANDES} ou mais pedidos no sorteio: {grandes.sum()}')
    print(f'na tabela, o centro deles fica a {br(afastamento(z, grandes), 1)} distâncias típicas do centro dos outros')
    print()

    if a.comparar:
        print(f'método              vizinhos mantidos  correlação das distâncias  afastamento dos {grandes.sum()}')
        for metodo in ('pca', 'kpca', 'tsne'):
            v, r, f = medidas(z, projetar(metodo, z, a)[0], grandes)
            print(f'{NOMES[metodo]:<20}{br(100 * v, 1) + "%":>17}  {br(r, 2):>25}  {br(f, 1):>16}')
        print()
        print(f'vizinhos mantidos: dos {VIZINHOS} clientes mais parecidos com cada um na tabela, quantos continuam '
              f'entre os {VIZINHOS} mais perto no desenho')
        print('correlação das distâncias: 1 quando a distância no desenho acompanha a distância na tabela')
        print(f'afastamento: a mesma conta da terceira linha, feita no desenho (na tabela: '
              f'{br(afastamento(z, grandes), 1)})')
        return

    if a.repetir:
        perp = a.perplexidade if a.perplexidade is not None else 30.0
        print(f'método: t-SNE, perplexidade {br(perp, 1).replace(",0", "")}, dois inícios sorteados (1 e 2), '
              f'{PASSOS} passos')
    else:
        print('método:', descricao(a.metodo, a, len(COLUNAS)))
    if a.repetir:
        desenhos = []
        for s in (1, 2):
            a.inicio, a.sorteio = 'aleatorio', s
            desenhos.append(projetar('tsne', z, a)[0])
            v, r, f = medidas(z, desenhos[-1], grandes)
            print(f'  início sorteado {s}: vizinhos mantidos {br(100 * v, 1)}%, correlação das distâncias '
                  f'{br(r, 2)}, afastamento dos {grandes.sum()}: {br(f, 1)}')
        d1, d2 = distancias2(desenhos[0]), distancias2(desenhos[1])
        print('  um desenho contra o outro:')
        print(f'    vizinhos no desenho 1 que são vizinhos no desenho 2: {br(100 * vizinhos_em_comum(d1, d2), 1)}%')
        print(f'    correlação entre as distâncias dos dois desenhos: {br(correlacao_distancias(d1, d2), 2)}')
        return

    novos_idx = np.array([], int)
    if a.mais:
        fora = np.setdiff1d(np.arange(len(x)), sorteados)
        novos_idx = np.sort(np.random.default_rng(2027).choice(fora, a.mais, replace=False))
    desenho, novos = projetar(a.metodo, z, a, z_todos[novos_idx] if a.mais else None)
    v, r, f = medidas(z, desenho, grandes)
    print(f'  vizinhos mantidos: {br(100 * v, 1)}% (dos {VIZINHOS} clientes mais parecidos com cada um na tabela, '
          f'quantos continuam entre os {VIZINHOS} mais perto no desenho)')
    print(f'  correlação das distâncias: {br(r, 2)} (1 quando a distância no desenho acompanha a da tabela)')
    print(f'  afastamento dos {grandes.sum()} clientes de {GRANDES} ou mais pedidos: {br(f, 1)} '
          f'(na tabela: {br(afastamento(z, grandes), 1)})')
    if a.metodo == 'kpca':
        g = a.gamma if a.gamma is not None else 1 / z.shape[1]
        semelhanca = np.exp(-g * distancias2(z))
        np.fill_diagonal(semelhanca, np.nan)
        print(f'  semelhança média com os outros clientes: {br(np.nanmean(semelhanca[grandes]), 3)} para um dos '
              f'{grandes.sum()}, {br(np.nanmean(semelhanca[~grandes]), 3)} para um cliente comum')

    if a.mais:
        print()
        print(f'chegam {a.mais} clientes novos, de fora do sorteio')
        if a.metodo != 'tsne':
            print(f'  o {NOMES[a.metodo]} tem a conta para pôr cada novo no desenho; os {a.clientes} antigos não se mexem')
            todos_z = np.r_[z, z_todos[novos_idx]]
            todos_d = np.r_[desenho, novos]
            juntos = np.r_[grandes, x[novos_idx, 0] >= GRANDES]
            v2, r2, _ = medidas(todos_z, todos_d, juntos)
            print(f'  com os novos no desenho: vizinhos mantidos {br(100 * v2, 1)}%, correlação das distâncias {br(r2, 2)}')
            if a.metodo == 'kpca':
                print(f'  para cada novo, a conta usa a semelhança dele com os {a.clientes} clientes do desenho')
        else:
            print('  o t-SNE não tem conta para um cliente novo: ele roda de novo com todos')
            todos = np.r_[sorteados, novos_idx]
            ordem = np.argsort(todos)
            refeito = projetar('tsne', z_todos[todos[ordem]], a)[0][np.argsort(ordem)][:a.clientes]
            d_antes, d_depois = distancias2(desenho), distancias2(refeito)
            print(f'  os {a.clientes} antigos, no desenho de antes e no refeito:')
            print(f'    vizinhos no desenho de antes que são vizinhos no refeito: '
                  f'{br(100 * vizinhos_em_comum(d_antes, d_depois), 1)}%')
            print(f'    correlação entre as distâncias dos dois desenhos: {br(correlacao_distancias(d_antes, d_depois), 2)}')
            antes = desenho / np.median(np.linalg.norm(desenho, axis=1))
            depois = refeito / np.median(np.linalg.norm(refeito, axis=1))
            print(f'    quanto cada antigo andou na tela (mediana, em distâncias típicas do centro): '
                  f'{br(np.median(np.linalg.norm(depois - antes, axis=1)), 2)}')

    if a.salvar:
        destino = Path(a.salvar)
        pd.DataFrame({'id_cliente': clientes['id_cliente'].to_numpy()[sorteados],
                      'x': desenho[:, 0].round(4), 'y': desenho[:, 1].round(4),
                      'grande': grandes.astype(int)}).to_csv(destino, index=False)
        print()
        print(f'gravei {destino} com {a.clientes} linhas: id_cliente, x, y e grande (1 = {GRANDES} ou mais pedidos)')


if __name__ == '__main__':
    try:
        main(sys.argv[1:])
    except Erro as e:
        print(f'erro: {e}', file=sys.stderr)
        sys.exit(2)
