"""Exercício: o plano de revisão de julho, com a nota do Igor em uso.

Rode de dentro da pasta desta aula:

    python3 meu_plano.py

Complete as três funções marcadas com FAÇA, seguindo o pedido do Marcos em pedido-julho.md.
O resto do programa aplica o seu plano às compras de abril a junho, como ensaio, e imprime o
relatório. Depois rode `python3 testar_meu_plano.py` para ver se o plano respeita o pedido.
"""
import pandas as pd

from plano import br, carregar, com_notas, estimar, revisar, wilson  # noqa: F401

SEMENTE = 20260702


def orcamento(dia):
    """Quantas compras a equipe revisa no dia, no formato '2026-04-04'.

    FAÇA: use o pedido. Dica de pandas: pd.Timestamp(dia).dayofweek é 0 na segunda e 6 no domingo.
    """
    raise NotImplementedError('complete orcamento(dia)')


def plano_julho(dia):
    """O plano de um dia: um dicionário que `revisar` entende, como os de avaliar.py.

    Chaves: 'topo' (quantas do topo), 'em_uso' (a nota que ordena o topo), 'sorteadas'
    (quantas sortear em cada faixa) e, se o plano tem faixa de comparação, 'candidata' e
    'k_candidata'. O total de revisões do dia não pode passar de orcamento(dia).

    FAÇA: decida o plano de segunda a sexta e o de sábado e domingo.
    """
    raise NotImplementedError('complete plano_julho(dia)')


def relatorio(revisadas, resposta, regra5):
    """As medidas que o Marcos pediu, calculadas só com as compras revisadas.

    revisadas: saída de revisar(), com as colunas 'faixa' e 'prob'.
    resposta: 1 ou 0 para cada compra revisada (a resposta da analista).
    regra5: True nas 5 compras de cada dia com a maior distância da entrega (todas as compras).

    FAÇA: retorne um dicionário com
      'alertas_uteis': fração de fraudes entre as revisadas do topo;
      'passaram': fraudes estimadas fora do topo, com os pesos;
      'regra_fora_do_topo': fraudes estimadas nas 5 da regra que ficaram fora do topo do Igor;
      'taxa_resto': (mínimo, máximo) da margem de 95% de Wilson para a taxa de fraude no resto.
    """
    raise NotImplementedError('complete relatorio(revisadas, resposta, regra5)')


def topo_de(novas, nota, k):
    ordem = novas.sort_values(['dia', nota, 'id_transacao'], ascending=[True, False, True])
    return (ordem.groupby('dia').cumcount() < k).reindex(novas.index)


def main(orcamento=orcamento, plano_julho=plano_julho, relatorio=relatorio):
    novas = com_notas(carregar())
    rev = revisar(novas, SEMENTE, plano_julho)
    resposta = novas.loc[rev.index, 'fraude']   # só a resposta de quem foi revisado
    por_dia = rev.groupby('dia').size()
    dias_uteis = [d for d in por_dia.index if orcamento(d) == 6]
    print('Ensaio do plano de julho com as compras de abril a junho')
    print(f'   revisões: {len(rev)} ({len(dias_uteis)} dias de semana, {len(por_dia) - len(dias_uteis)} de fim de semana)')
    print(f'   por faixa: {rev.faixa.value_counts().to_dict()}')
    print(f'   fraudes por faixa: {resposta.groupby(rev.faixa).sum().astype(int).to_dict()}')
    r = relatorio(rev, resposta, topo_de(novas, 'regra', 5))
    fora = len(novas) - int((rev.faixa == 'topo').sum())
    print(f'   alertas úteis do topo: {br(100 * r["alertas_uteis"])}%')
    print(f'   fraudes estimadas fora do topo: {br(r["passaram"])}')
    print(f'   fraudes estimadas nas 5 da regra fora do topo: {br(r["regra_fora_do_topo"])}')
    baixo, alto = r['taxa_resto']
    print(f'   taxa no resto, margem de 95%: de {br(100 * baixo, 2)}% a {br(100 * alto, 2)}%; '
          f'compras fora do topo: {fora}')


if __name__ == '__main__':
    main()
