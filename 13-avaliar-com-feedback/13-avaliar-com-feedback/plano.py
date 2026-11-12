"""Peças do plano de revisão: as duas notas, o sorteio por faixa e a conta com pesos.

Os outros programas dessa pasta importam daqui. Rode tudo de dentro da pasta da aula.

A regra da equipe ordena as compras do dia pela distância entre a entrega e o cadastro.
A nota do Igor soma o quanto cada compra se afasta do normal em três colunas: valor, idade da
conta e distância. O normal é medido com as compras de janeiro a março, sem a coluna `fraude`.
"""
import math

import numpy as np
import pandas as pd

DADOS = '../dados/fraude/transacoes.csv'
INICIO = '2026-04-01'
COLUNAS_IGOR = ['valor', 'idade_conta_dias', 'distancia_entrega_km']


def carregar(caminho=DADOS):
    compras = pd.read_csv(caminho)
    compras['dia'] = compras['data_hora'].str[:10]
    return compras


def com_notas(compras):
    """Compras de abril em diante, com as colunas `regra` e `igor`.

    `regra` é a distância da entrega. `igor` é a soma, nas três colunas, da distância até a
    mediana de janeiro a março, em escala log e dividida pelo desvio típico (MAD × 1,4826).
    """
    referencia = compras[compras['dia'] < INICIO]
    novas = compras[compras['dia'] >= INICIO].copy()
    novas['regra'] = novas['distancia_entrega_km']
    nota = 0
    for c in COLUNAS_IGOR:
        x = np.log10(referencia[c] + 1)
        mediana = x.median()
        desvio = 1.4826 * (x - mediana).abs().median()
        nota = nota + (np.log10(novas[c] + 1) - mediana).abs() / desvio
    novas['igor'] = nota.round(3)
    return novas


def ordem_do_dia(do_dia, nota):
    """Compras de um dia da maior nota para a menor; empate: a compra mais antiga primeiro."""
    return do_dia.sort_values([nota, 'id_transacao'], ascending=[False, True])


def sortear_na_faixa(faixa, quantas, sorteio, nome):
    """Sorteia `quantas` compras da faixa, sem repetir, e anota a probabilidade de cada uma.

    Cada compra da faixa tem a mesma chance: quantas / tamanho da faixa. O peso de uma compra
    revisada é o inverso disso: quantas compras da faixa ela representa.
    """
    if quantas < 0:
        raise ValueError('quantas não pode ser negativo')
    k = min(quantas, len(faixa))
    if k == 0:
        return faixa.head(0).assign(faixa=nome, prob=1.0)
    escolha = sorted(sorteio.choice(len(faixa), size=k, replace=False))
    return faixa.iloc[escolha].assign(faixa=nome, prob=k / len(faixa))


def faixas_do_dia(do_dia, topo=5, em_uso='regra', candidata=None, k_candidata=5):
    """Divide as compras de um dia em faixas, só pelas notas.

    topo: as `topo` primeiras da nota em uso.
    candidata: se tem uma nota em teste, as `k_candidata` primeiras dela que ficaram fora do topo.
    resto: as outras compras do dia.
    """
    ordem = ordem_do_dia(do_dia, em_uso)
    faixas = {'topo': ordem.head(topo)}
    fora = ordem.iloc[topo:]
    if candidata is not None:
        da_candidata = ordem_do_dia(do_dia, candidata).head(k_candidata).index
        faixas['candidata'] = fora[fora.index.isin(da_candidata)]
        fora = fora[~fora.index.isin(da_candidata)]
    faixas['resto'] = fora
    return faixas


def sortear(faixas, sorteio, sorteadas):
    """As compras revisadas: o topo inteiro e, em cada faixa, as sorteadas que o plano pede.

    faixas: lista com as faixas de cada dia. sorteadas: quantas sortear em cada faixa, por
    exemplo {'candidata': 1, 'resto': 1}, ou uma lista com um desses por dia.
    """
    partes = []
    for i, dia in enumerate(faixas):
        pedido = sorteadas[i] if isinstance(sorteadas, list) else sorteadas
        partes.append(dia['topo'].assign(faixa='topo', prob=1.0))
        for nome, quantas in pedido.items():
            if nome in dia:
                partes.append(sortear_na_faixa(dia[nome], quantas, sorteio, nome))
    return pd.concat(partes)


def montar_faixas(novas, plano_do_dia):
    """Faixas e sorteadas de cada dia. A coluna `fraude`, se existir, não entra.

    plano_do_dia(dia) retorna um dicionário com `topo`, `sorteadas` e, se quiser, `em_uso`,
    `candidata` e `k_candidata`.
    """
    sem_resposta = novas.drop(columns='fraude', errors='ignore')
    faixas, sorteadas = [], []
    for dia, do_dia in sem_resposta.groupby('dia', sort=True):
        plano = dict(plano_do_dia(dia))
        sorteadas.append(plano.pop('sorteadas'))
        faixas.append(faixas_do_dia(do_dia, **plano))
    return faixas, sorteadas


def revisar(novas, semente, plano_do_dia):
    """Aplica o plano a cada dia, com um sorteio que sai igual para a mesma semente."""
    faixas, sorteadas = montar_faixas(novas, plano_do_dia)
    return sortear(faixas, np.random.default_rng(semente), sorteadas)


def estimar(revisadas, resposta, dentro=None):
    """Quantas fraudes há num conjunto de compras, contando só as revisadas, cada uma com seu peso.

    revisadas: saída de `revisar`, com a coluna `prob`.
    resposta: Series com 1 (fraude) ou 0 para cada compra revisada, pelo índice.
    dentro: Series de True/False que diz quais compras formam o conjunto; sem ela, todas.
    """
    if revisadas['prob'].isna().any() or not revisadas['prob'].between(0, 1, inclusive='right').all():
        raise ValueError('toda compra revisada precisa de probabilidade maior que 0 e no máximo 1')
    falta = revisadas.index.difference(resposta.dropna().index)
    if len(falta):
        raise ValueError(f'{len(falta)} compra(s) revisada(s) sem resposta da analista')
    r = revisadas if dentro is None else revisadas[dentro.reindex(revisadas.index, fill_value=False)]
    return float((resposta.loc[r.index] / r['prob']).sum())


def simular(faixas, sorteadas, resposta, conjuntos, vezes, semente):
    """Repete o sorteio `vezes` vezes e guarda a estimativa de cada conjunto em cada repetição.

    Serve para planejar: com o gabarito de um período que já tem as contestações, eu vejo quanto
    a estimativa varia de um sorteio para outro. conjuntos: nome -> Series de True/False.
    Retorna um DataFrame com uma linha por repetição e uma coluna por conjunto.
    """
    sorteio = np.random.default_rng(semente)
    marcas = {n: c.to_numpy() for n, c in conjuntos.items()}
    pos = pd.Series(np.arange(len(resposta)), index=resposta.index)
    y = resposta.to_numpy()
    fixas = {n: 0.0 for n in conjuntos}
    blocos = []  # (posições da faixa, quantas)
    for i, dia in enumerate(faixas):
        p = pos.loc[dia['topo'].index].to_numpy()
        for n, m in marcas.items():
            fixas[n] += float((y[p] * m[p]).sum())
        pedido = sorteadas[i] if isinstance(sorteadas, list) else sorteadas
        for nome, quantas in pedido.items():
            if nome in dia and len(dia[nome]) and quantas:
                blocos.append((pos.loc[dia[nome].index].to_numpy(), min(quantas, len(dia[nome]))))
    linhas = []
    for _ in range(vezes):
        soma = dict(fixas)
        for p, k in blocos:
            e = sorteio.choice(p, size=k, replace=False)
            peso = len(p) / k
            for n, m in marcas.items():
                soma[n] += peso * float((y[e] * m[e]).sum())
        linhas.append(soma)
    return pd.DataFrame(linhas)


def wilson(achadas, revisadas, z=1.959964):
    """Intervalo de 95% de Wilson para uma proporção, como em Estatística para decisão."""
    if revisadas <= 0:
        raise ValueError('sem compra revisada não há proporção')
    p = achadas / revisadas
    den = 1 + z * z / revisadas
    centro = (p + z * z / (2 * revisadas)) / den
    meia = z * math.sqrt(p * (1 - p) / revisadas + z * z / (4 * revisadas ** 2)) / den
    return max(0.0, centro - meia), min(1.0, centro + meia)


def br(x, casas=1):
    """Número com vírgula decimal e ponto de milhar."""
    texto = f'{x:,.{casas}f}'
    return texto.replace(',', '_').replace('.', ',').replace('_', '.')
