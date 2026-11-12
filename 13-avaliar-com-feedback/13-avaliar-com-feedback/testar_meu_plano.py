"""Testes do exercício: o plano de julho respeita o pedido e as contas usam os pesos.

Rode de dentro da pasta desta aula, depois de completar meu_plano.py:

    python3 testar_meu_plano.py

Para testar outro arquivo com as mesmas três funções: python3 testar_meu_plano.py outro.py
Sai com código 0 quando tudo passa e 1 quando algum teste falha.
"""
import importlib.util
import math
import sys

import numpy as np
import pandas as pd

from plano import carregar, com_notas, faixas_do_dia, sortear_na_faixa

falhas = []


def teste(nome, funcao):
    try:
        funcao()
        print(f'ok     {nome}')
    except NotImplementedError as e:
        falhas.append(nome)
        print(f'FALTA  {nome}: {e}')
    except Exception as e:  # noqa: BLE001 - o teste mostra qualquer erro do aluno
        falhas.append(nome)
        print(f'FALHA  {nome}: {type(e).__name__}: {e}')


def carregar_modulo(caminho):
    spec = importlib.util.spec_from_file_location('plano_do_aluno', caminho)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


m = carregar_modulo(sys.argv[1] if len(sys.argv) > 1 else 'meu_plano.py')
novas = com_notas(carregar()).drop(columns='fraude')
dias = sorted(novas['dia'].unique())


def t_orcamento():
    # 2026-04-03 é sexta; 04 e 05 são sábado e domingo; 06 é segunda.
    obtido = [m.orcamento(d) for d in ['2026-04-03', '2026-04-04', '2026-04-05', '2026-04-06']]
    assert obtido == [6, 3, 3, 6], f'esperado [6, 3, 3, 6], veio {obtido}'


def planos():
    for d in dias:
        p = dict(m.plano_julho(d))
        sorteadas = p.pop('sorteadas')
        yield d, p, sorteadas, faixas_do_dia(novas[novas['dia'] == d], **p)


def t_igor_no_topo():
    errados = [d for d, p, _, _ in planos() if p.get('em_uso') != 'igor']
    assert not errados, f'o topo precisa ser da nota do Igor; dias com outra nota: {errados[:3]}'


def t_cabe_no_dia():
    passou = []
    for d, _, sorteadas, faixas in planos():
        total = len(faixas['topo']) + sum(min(q, len(faixas.get(n, []))) for n, q in sorteadas.items())
        if total > m.orcamento(d):
            passou.append((d, total))
    assert not passou, f'dias com mais revisões que a equipe faz: {passou[:3]}'


def t_toda_compra_tem_chance():
    sem_chance = []
    for d, _, sorteadas, faixas in planos():
        for nome, faixa in faixas.items():
            if nome != 'topo' and len(faixa) and sorteadas.get(nome, 0) < 1:
                sem_chance.append((d, nome, len(faixa)))
    assert not sem_chance, ('toda faixa fora do topo precisa de pelo menos 1 sorteada por dia; '
                            f'faixas sem sorteio: {sem_chance[:3]}')


def t_pesos_somam_a_faixa():
    sorteio = np.random.default_rng(1)
    for d, _, sorteadas, faixas in list(planos())[:14]:
        for nome, quantas in sorteadas.items():
            if nome in faixas and len(faixas[nome]):
                s = sortear_na_faixa(faixas[nome], quantas, sorteio, nome)
                assert math.isclose((1 / s['prob']).sum(), len(faixas[nome])), (d, nome)


def caso(fraude_no_resto):
    """Um dia feito à mão: 4 no topo (1 fraude), 1 sorteada de 3 na faixa da regra (fraude),
    1 sorteada de 40 no resto. A compra 5 está nas 5 da regra; a 6 não."""
    revisadas = pd.DataFrame({'faixa': ['topo'] * 4 + ['candidata', 'resto'],
                              'prob': [1, 1, 1, 1, 1 / 3, 1 / 40]}, index=[1, 2, 3, 4, 5, 6])
    resposta = pd.Series([1, 0, 0, 0, 1, int(fraude_no_resto)], index=revisadas.index)
    regra5 = pd.Series(False, index=range(1, 51))
    regra5.loc[[1, 2, 5, 30, 31]] = True
    return revisadas, resposta, regra5


def t_relatorio_com_pesos():
    r = m.relatorio(*caso(False))
    assert math.isclose(r['alertas_uteis'], 0.25), f'alertas úteis: esperado 0,25, veio {r["alertas_uteis"]}'
    assert math.isclose(r['passaram'], 3), f'passaram: a sorteada de 3 vale 3 compras; esperado 3, veio {r["passaram"]}'
    assert math.isclose(r['regra_fora_do_topo'], 3), f'regra fora do topo: esperado 3, veio {r["regra_fora_do_topo"]}'
    baixo, alto = r['taxa_resto']
    assert baixo == 0 and 0.79 < alto < 0.80, f'Wilson de 0 em 1: esperado de 0 a 0,79; veio {r["taxa_resto"]}'
    r = m.relatorio(*caso(True))
    assert math.isclose(r['passaram'], 43), f'passaram com fraude no resto: esperado 3 + 40 = 43, veio {r["passaram"]}'
    assert math.isclose(r['regra_fora_do_topo'], 3), 'a compra 6 não está nas 5 da regra e não entra nessa conta'


def t_recusa_revisada_sem_resposta():
    revisadas, resposta, regra5 = caso(False)
    try:
        m.relatorio(revisadas, resposta.drop(6), regra5)
    except ValueError:
        return
    raise AssertionError('uma compra revisada sem resposta precisa dar erro, não virar "não é fraude"')


for nome, f in [('orçamento: 6 de segunda a sexta, 3 no sábado e no domingo', t_orcamento),
                ('o topo é da nota do Igor', t_igor_no_topo),
                ('nenhum dia passa do orçamento', t_cabe_no_dia),
                ('toda compra fora do topo tem chance de ser sorteada, todo dia', t_toda_compra_tem_chance),
                ('os pesos de cada faixa somam o tamanho dela', t_pesos_somam_a_faixa),
                ('relatório conta com os pesos', t_relatorio_com_pesos),
                ('relatório recusa compra revisada sem resposta', t_recusa_revisada_sem_resposta)]:
    teste(nome, f)

print(f'{7 - len(falhas)} de 7 testes passaram')
sys.exit(1 if falhas else 0)
