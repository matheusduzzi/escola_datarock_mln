"""Avaliar a fila com o retorno da equipe: o que dá para dizer, a conta errada e o plano.

Rode de dentro da pasta desta aula:

    python3 avaliar.py

As partes 1 a 4 só usam a resposta das compras que a equipe revisou. A coluna `fraude` inteira
entra na parte 5, como se as contestações do banco já tivessem chegado.
"""
import pandas as pd

from plano import br, carregar, com_notas, estimar, montar_faixas, revisar, simular, wilson

SEMENTE = 20260701   # o sorteio de abril a junho sai igual toda vez
DIA_EXEMPLO = '2026-05-12'
VEZES = 1000


def hoje(dia):
    """Como a equipe trabalhou de abril a junho: 5 do topo da regra e 1 sorteada no resto."""
    return {'topo': 5, 'sorteadas': {'resto': 1}}


def ensaio(dia):
    """O plano para testar a nota do Igor sem passar das 6 revisões do dia."""
    return {'topo': 4, 'candidata': 'igor', 'k_candidata': 5,
            'sorteadas': {'candidata': 1, 'resto': 1}}


def topo_de(novas, nota, k):
    """True nas k compras de cada dia com a nota mais alta (empate: a mais antiga)."""
    ordem = novas.sort_values(['dia', nota, 'id_transacao'], ascending=[True, False, True])
    return (ordem.groupby('dia').cumcount() < k).reindex(novas.index)


def pct(a, b, casas=1):
    return br(100 * a / b, casas) + '%'


def main():
    novas = com_notas(carregar())
    fraude = novas['fraude']   # só lida nas compras revisadas até a parte 5
    igor5 = topo_de(novas, 'igor', 5)

    rev = revisar(novas, SEMENTE, hoje)
    resp = fraude.loc[rev.index]   # a resposta da analista, só para quem ela revisou
    topo, sorte = rev[rev.faixa == 'topo'], rev[rev.faixa == 'resto']
    fora = len(novas) - len(topo)
    achadas_topo, achadas_sorte = int(resp[topo.index].sum()), int(resp[sorte.index].sum())
    baixo, alto = wilson(achadas_sorte, len(sorte))
    print('1. Abril a junho com a regra: 5 do topo e 1 sorteada por dia')
    print(f'   dias: {novas.dia.nunique()}; compras: {len(novas)}; revisões: {len(rev)}')
    print(f'   topo: {len(topo)} revisadas, {achadas_topo} fraudes; alertas úteis: {pct(achadas_topo, len(topo))}')
    print(f'   sorteadas: {len(sorte)} revisadas, {achadas_sorte} fraudes')
    print(f'   compras fora do topo: {fora}; estimativa de fraudes que passaram: '
          f'{br(estimar(rev, resp, rev.faixa != "topo"))}')
    print(f'   margem de 95% (Wilson) para a taxa fora do topo: de {pct(baixo, 1, 2)} a {pct(alto, 1, 2)}, '
          f'ou de {round(fora * baixo)} a {round(fora * alto)} fraudes')

    revisadas_igor = rev[igor5.reindex(rev.index)]
    print('2. A conta do Igor: a nota dele só nas compras que a equipe revisou')
    print(f'   compras no topo 5 do Igor: {int(igor5.sum())}; dessas, revisadas: {len(revisadas_igor)} '
          f'({int((revisadas_igor.faixa == "topo").sum())} do topo da regra, '
          f'{int((revisadas_igor.faixa == "resto").sum())} sorteadas)')
    print(f'   fraudes entre as revisadas do topo do Igor: {int(resp[revisadas_igor.index].sum())}; '
          f'a regra achou {achadas_topo}')
    print(f'   compras do topo do Igor que ninguém revisou: {int(igor5.sum()) - len(revisadas_igor)}')

    rev2 = revisar(novas, SEMENTE, ensaio)
    resp2 = fraude.loc[rev2.index]
    exemplo = rev2[rev2.dia == DIA_EXEMPLO]
    faixas_ex = montar_faixas(novas[novas.dia == DIA_EXEMPLO], ensaio)[0][0]
    print(f'3. Um dia do ensaio, {DIA_EXEMPLO}: 4 do topo da regra, 1 sorteada entre as 5 do Igor, 1 no resto')
    print(f'   compras no dia: {int((novas.dia == DIA_EXEMPLO).sum())}; '
          f'faixas: topo {len(faixas_ex["topo"])}, candidata {len(faixas_ex["candidata"])}, resto {len(faixas_ex["resto"])}')
    print('   id     faixa      chance   peso  distância km   igor  fraude')
    for i, l in exemplo.iterrows():
        print(f'   {l.id_transacao:<5}  {l.faixa:<9}  {br(l.prob, 3):>6}  {br(1 / l.prob, 0):>5}  '
              f'{br(l.distancia_entrega_km):>12}  {br(l.igor, 2):>5}  {int(resp2[i]):>6}')

    t2 = rev2[rev2.faixa == 'topo']
    achadas2 = int(resp2[t2.index].sum())
    print('4. O ensaio inteiro, abril a junho, com as mesmas 6 revisões por dia')
    for nome in ['topo', 'candidata', 'resto']:
        f = rev2[rev2.faixa == nome]
        print(f'   {nome:<9}: {len(f)} revisadas, {int(resp2[f.index].sum())} fraudes')
    print(f'   alertas úteis da regra (4 do topo): {pct(achadas2, len(t2))}')
    est_igor = estimar(rev2, resp2, igor5)
    print(f'   fraudes estimadas no topo 5 do Igor: {br(est_igor)}; alertas úteis estimados: '
          f'{pct(est_igor, int(igor5.sum()))}')
    print(f'   fraudes estimadas fora do topo da regra: {br(estimar(rev2, resp2, rev2.faixa != "topo"))}')
    print(f'   fraudes que a equipe achou no ensaio: {int(resp2.sum())}; com a regra de hoje: '
          f'{achadas_topo + achadas_sorte}')

    regra5, regra4 = topo_de(novas, 'regra', 5), topo_de(novas, 'regra', 4)
    print('5. Quando todas as contestações chegam (a coluna fraude inteira entra aqui)')
    print(f'   fraudes de abril a junho: {int(fraude.sum())}')
    print(f'   no topo 5 da regra: {int(fraude[regra5].sum())}; no topo 4: {int(fraude[regra4].sum())}; '
          f'no topo 5 do Igor: {int(fraude[igor5].sum())}')
    print(f'   passaram fora do topo 5 da regra: {int(fraude[~regra5].sum())}; '
          f'fora do topo 4: {int(fraude[~regra4].sum())}')

    print(f'6. O mesmo sorteio repetido {VEZES} vezes, com o gabarito da parte 5')
    print('   plano               o que estimar             real  mediana   90% das vezes   saiu 0')
    for nome, plano, regra in [('5 + 1 (hoje)', hoje, regra5), ('4 + 1 + 1 (ensaio)', ensaio, regra4)]:
        faixas, sorteadas = montar_faixas(novas, plano)
        sim = simular(faixas, sorteadas, fraude, {'passaram': ~regra, 'igor5': igor5}, VEZES, SEMENTE)
        for col, rotulo, real in [('passaram', 'fraudes fora do topo', int(fraude[~regra].sum())),
                                  ('igor5', 'fraudes no topo do Igor', int(fraude[igor5].sum()))]:
            s = sim[col]
            print(f'   {nome:<18}  {rotulo:<24}  {real:>4}  {br(s.median(), 0):>7}  '
                  f'{br(s.quantile(0.05), 0):>5} a {br(s.quantile(0.95), 0):<5}  {pct((s == 0).sum(), len(s), 0):>6}')


if __name__ == '__main__':
    pd.set_option('display.width', 120)
    main()
