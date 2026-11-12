"""Nota de estranheza de cada compra: a soma da raridade de cada coluna.

É a ideia do HBOS (histogram-based outlier score, Goldstein e Dengel, 2012). Para cada coluna,
eu monto um histograma com as compras de referência. A altura de cada barra é dividida pela da
barra mais alta, então a faixa mais comum vale 1. A raridade de uma compra nessa coluna é
log10(1 / altura): 0 na faixa mais comum, 1 numa faixa dez vezes menos comum, 2 numa cem vezes
menos comum. A nota é a soma das raridades.

A coluna `fraude` não entra aqui. Ela só chega meses depois, quando o banco aceita a contestação.
"""
import numpy as np
import pandas as pd

# Colunas contínuas: 10 faixas de mesma largura, na escala que a função indica.
CONTINUAS = {
    'valor': 'log',
    'idade_conta_dias': 'log',
    'distancia_entrega_km': 'log',
}
# Colunas com poucos valores: cada valor é uma faixa.
DISCRETAS = ['itens', 'tentativas_cartao', 'hora', 'canal']
COLUNAS = list(CONTINUAS) + DISCRETAS
FAIXAS = 10


def _escala(x, escala):
    x = np.asarray(x, dtype=float)
    return np.log10(x + 1) if escala == 'log' else x


def ajustar(referencia, faixas=FAIXAS, escala=None):
    """Monta os histogramas a partir das compras de referência.

    `escala='linear'` troca a escala das colunas contínuas, para comparar.
    """
    faltando = [c for c in COLUNAS if c not in referencia.columns]
    if faltando:
        raise ValueError(f'faltam colunas na referência: {faltando}')
    if len(referencia) == 0:
        raise ValueError('a referência está vazia')
    n = len(referencia)
    piso = 0.5 / n  # valor que não apareceu na referência conta como meia compra
    modelo = {'n': n, 'piso': piso, 'colunas': {}}
    for col in CONTINUAS:
        esc = escala or CONTINUAS[col]
        t = _escala(referencia[col], esc)
        bordas = np.linspace(t.min(), t.max(), faixas + 1)
        contagem, _ = np.histogram(t, bordas)
        fracao = contagem / n
        modelo['colunas'][col] = {'tipo': 'continua', 'escala': esc, 'bordas': bordas,
                                  'fracao': fracao, 'maior': fracao.max()}
    for col in DISCRETAS:
        fracao = referencia[col].value_counts(normalize=True)
        modelo['colunas'][col] = {'tipo': 'discreta', 'fracao': fracao, 'maior': fracao.max()}
    return modelo


def pontuar(modelo, compras):
    """Retorna uma tabela com a raridade de cada coluna e a nota (soma), no índice de `compras`."""
    partes = {}
    for col, m in modelo['colunas'].items():
        if m['tipo'] == 'continua':
            t = _escala(compras[col], m['escala'])
            bordas = m['bordas']
            i = np.clip(np.searchsorted(bordas, t, side='right') - 1, 0, len(m['fracao']) - 1)
            fracao = np.where((t < bordas[0]) | (t > bordas[-1]), 0.0, m['fracao'][i])
        else:
            fracao = compras[col].map(m['fracao']).fillna(0.0).to_numpy(dtype=float)
        fracao = np.maximum(fracao, modelo['piso'])
        partes[col] = np.log10(m['maior'] / fracao)
    tabela = pd.DataFrame(partes, index=compras.index)
    tabela['nota'] = tabela[COLUNAS].sum(axis=1)
    return tabela


def motivos(linha, quantos=2):
    """As colunas que mais pesaram na nota de uma compra, da maior para a menor."""
    return list(linha[COLUNAS].astype(float).sort_values(ascending=False).index[:quantos])
