"""Isolation Forest, LOF e One-Class SVM nas compras com cartão da Loja Exemplo.

    python3 compras.py                                 # os três métodos com os parâmetros padrão
    python3 compras.py --vizinhos 50                   # LOF com 50 vizinhos
    python3 compras.py --nu 0.2 --gamma 0.05           # One-Class SVM com outro nu e outro gamma
    python3 compras.py --sorteio 3                     # outro sorteio das árvores do Isolation Forest
    python3 compras.py --sem-escala                    # sem a escala z nas colunas
    python3 compras.py --referencia-sem-contestadas    # tira da referência as fraudes de janeiro a março

Cada método aprende o que é comum com as compras de janeiro a março e dá uma nota às compras de
abril a junho. A coluna `fraude` não entra em nenhuma nota. Ela só volta no fim, para contar
quantas fraudes caíram nas 5 compras de nota mais alta de cada dia, a régua da fila do Marcos.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

CSV = Path('..') / 'dados' / 'fraude' / 'transacoes.csv'
INICIO_NOTA = '2026-04-01'
DIA_EXEMPLO = '2026-04-21'
TOPO = 5
COLUNAS = ['valor', 'itens', 'idade_conta_dias', 'tentativas_cartao', 'distancia_entrega_km', 'hora', 'canal']
METODOS = ['Isolation Forest', 'LOF', 'One-Class SVM']


def fmt(x, casas=2):
    return f'{x:,.{casas}f}'.replace(',', 'X').replace('.', ',').replace('X', '.')


def numero(texto):
    return float(str(texto).replace(',', '.'))


def carregar():
    if not CSV.is_file():
        sys.exit('erro: não achei ../dados/fraude/transacoes.csv. Rode de dentro da pasta da aula, '
                 '11-isolation-forest-lof-e-one-class-svm/.')
    compras = pd.read_csv(CSV, parse_dates=['data_hora'])
    if compras[COLUNAS].isna().any().any():
        sys.exit('erro: há valores vazios nas colunas da nota')
    return compras


def colunas_da_nota(compras, com_log=True):
    """As sete colunas que a analista vê na tela. Valor, idade da conta e distância entram em log."""
    x = pd.DataFrame(index=compras.index)
    for c in ['valor', 'idade_conta_dias', 'distancia_entrega_km']:
        x[c] = np.log10(compras[c] + 1) if com_log else compras[c]
    for c in ['itens', 'tentativas_cartao', 'hora']:
        x[c] = compras[c].astype(float)
    x['canal'] = (compras['canal'] == 'app').astype(float)
    return x


def notas(ref, novas, vizinhos, nu, gamma, sorteio, escala):
    """Ajusta cada método na referência e dá nota às compras novas: maior quer dizer mais estranha."""
    xr, xn = ref.to_numpy(), novas.to_numpy()
    if escala:
        z = StandardScaler().fit(xr)
        xr, xn = z.transform(xr), z.transform(xn)
    floresta = IsolationForest(random_state=sorteio).fit(xr)
    lof = LocalOutlierFactor(n_neighbors=vizinhos, novelty=True).fit(xr)
    svm = OneClassSVM(nu=nu, gamma=gamma).fit(xr)
    return {
        'Isolation Forest': -floresta.score_samples(xn),
        'LOF': -lof.score_samples(xn),
        'One-Class SVM': -svm.score_samples(xn),
    }


def topo_do_dia(compras, nota, topo=TOPO):
    """As `topo` compras de nota mais alta de cada dia. Empate: entra a compra mais antiga."""
    tabela = compras[['id_transacao', 'dia']].assign(nota=nota)
    tabela = tabela.sort_values(['dia', 'nota', 'id_transacao'], ascending=[True, False, True])
    tabela['pos'] = tabela.groupby('dia').cumcount() + 1
    return tabela[tabela['pos'] <= topo]


def main(argv=None):
    p = argparse.ArgumentParser(description='Três notas de estranheza nas compras de abril a junho.')
    p.add_argument('--vizinhos', default='20', help='vizinhos do LOF (n_neighbors), padrão 20')
    p.add_argument('--nu', default='0.05', help='nu do One-Class SVM, entre 0 e 1, padrão 0,05')
    p.add_argument('--gamma', default='scale', help="gamma do One-Class SVM: um número maior que 0 ou 'scale' (padrão)")
    p.add_argument('--sorteio', default='0', help='número do sorteio das árvores do Isolation Forest, padrão 0')
    p.add_argument('--sem-escala', action='store_true', help='não põe as colunas na escala z')
    p.add_argument('--referencia-sem-contestadas', action='store_true',
                   help='tira da referência as compras de janeiro a março que tiveram contestação de fraude')
    a = p.parse_args(argv)
    try:
        vizinhos = int(a.vizinhos)
        nu = numero(a.nu)
        gamma = a.gamma if a.gamma == 'scale' else numero(a.gamma)
        sorteio = int(a.sorteio)
    except ValueError:
        p.error('--vizinhos e --sorteio são números inteiros; --nu e --gamma são números (ou --gamma scale)')
    if not 1 <= vizinhos <= 500:
        p.error('--vizinhos vai de 1 a 500')
    if not 0 < nu <= 1:
        p.error('--nu precisa ser maior que 0 e no máximo 1')
    if gamma != 'scale' and gamma <= 0:
        p.error('--gamma precisa ser maior que 0')

    compras = carregar()
    compras['dia'] = compras['data_hora'].dt.strftime('%Y-%m-%d')
    fraude = compras.pop('fraude')  # a contestação chega meses depois: fica fora de tudo até a parte 3
    eh_ref = compras['data_hora'] < INICIO_NOTA
    if a.referencia_sem_contestadas:
        eh_ref = eh_ref & (fraude == 0)
    ref, novas = compras[eh_ref], compras[compras['data_hora'] >= INICIO_NOTA].reset_index(drop=True)
    fraude_novas = fraude[compras['data_hora'] >= INICIO_NOTA].reset_index(drop=True)

    print('1. Referência e compras com nota')
    print(f'   referência: compras de 2026-01-01 a 2026-03-31: {len(ref)}'
          + (' (sem as contestadas de janeiro a março)' if a.referencia_sem_contestadas else ''))
    print(f'   com nota: compras de {INICIO_NOTA} a 2026-06-30: {len(novas)} em {novas["dia"].nunique()} dias')
    print(f'   colunas: {", ".join(COLUNAS)}; valor, idade da conta e distância em log; '
          + ('sem escala z' if a.sem_escala else 'todas na escala z da referência'))
    print(f'   LOF: {vizinhos} vizinhos; One-Class SVM: nu {fmt(nu)}, gamma '
          f'{gamma if gamma == "scale" else fmt(gamma)}; Isolation Forest: sorteio {sorteio}')

    x_ref, x_novas = colunas_da_nota(ref), colunas_da_nota(novas)
    resultado = notas(x_ref, x_novas, vizinhos, nu, gamma, sorteio, escala=not a.sem_escala)

    print(f'2. A nota de cada método (score_samples com o sinal trocado: maior = mais estranha)')
    print(f'   {"método":<17}{"menor":>9}{"mediana":>9}{"maior":>9}')
    for m in METODOS:
        n = resultado[m]
        print(f'   {m:<17}{fmt(n.min()):>9}{fmt(np.median(n)):>9}{fmt(n.max()):>9}')

    conta_nova = novas['idade_conta_dias'] <= 30
    total = int(fraude_novas.sum())
    print(f'3. Quando a contestação chega: fraudes nas {TOPO} do topo de cada dia ({TOPO * novas["dia"].nunique()} revisões)')
    print(f'   fraudes de abril a junho: {total} ({int((fraude_novas & conta_nova).sum())} de conta nova, até 30 dias; '
          f'{int((fraude_novas & ~conta_nova).sum())} de conta antiga)')
    print(f'   {"método":<17}{"no topo":>8}{"conta nova":>12}{"conta antiga":>14}{"na 1ª do dia":>14}')
    topos = {}
    for m in METODOS:
        t = topo_do_dia(novas, resultado[m])
        topos[m] = t
        f = fraude_novas.loc[t.index] == 1
        nova = conta_nova.loc[t.index]
        primeira = int((f & (t['pos'] == 1)).sum())
        print(f'   {m:<17}{int(f.sum()):>8}{int((f & nova).sum()):>12}{int((f & ~nova).sum()):>14}{primeira:>14}')

    print('4. O que mais aparece nas compras do topo que não eram fraude')
    liberadas = {m: novas.loc[t.index][fraude_novas.loc[t.index] == 0] for m, t in topos.items()}
    print(f'   {"método":<17}{"compras":>8}{"> 50 km":>9}{"0 a 5 h":>9}{"3+ tent.":>10}{"8+ itens":>10}'
          f'{"conta até 30 d":>16}')
    for m in METODOS:
        lib = liberadas[m]
        print(f'   {m:<17}{len(lib):>8}{int((lib["distancia_entrega_km"] > 50).sum()):>9}'
              f'{int((lib["hora"] <= 5).sum()):>9}{int((lib["tentativas_cartao"] >= 3).sum()):>10}'
              f'{int((lib["itens"] >= 8).sum()):>10}{int((lib["idade_conta_dias"] <= 30).sum()):>16}')
    todas = novas[fraude_novas == 0]
    print(f'   {"todas as liberadas":<17}{len(todas):>8}{int((todas["distancia_entrega_km"] > 50).sum()):>9}'
          f'{int((todas["hora"] <= 5).sum()):>9}{int((todas["tentativas_cartao"] >= 3).sum()):>10}'
          f'{int((todas["itens"] >= 8).sum()):>10}{int((todas["idade_conta_dias"] <= 30).sum()):>16}')

    print(f'5. Os três topos se parecem? (compras em comum nas {TOPO * novas["dia"].nunique()} revisões)')
    pares = [('Isolation Forest', 'LOF'), ('Isolation Forest', 'One-Class SVM'), ('LOF', 'One-Class SVM')]
    for m1, m2 in pares:
        comum = len(set(topos[m1]['id_transacao']) & set(topos[m2]['id_transacao']))
        print(f'   {m1} e {m2}: {comum}')
    print(f'   nas três: {len(set.intersection(*(set(t["id_transacao"]) for t in topos.values())))}')

    print(f'6. A fila de {DIA_EXEMPLO} pelos três métodos (ids, da nota mais alta para a mais baixa)')
    dia = novas['dia'] == DIA_EXEMPLO
    for m in METODOS:
        t = topos[m]
        ids = t[t['dia'] == DIA_EXEMPLO]['id_transacao'].tolist()
        print(f'   {m:<17}{", ".join(map(str, ids))}')
    contestadas = novas[dia & (fraude_novas == 1)]['id_transacao'].tolist()
    print(f'   contestadas depois, desse dia: {", ".join(map(str, contestadas)) or "nenhuma"} '
          f'(de {int(dia.sum())} compras)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
