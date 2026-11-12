"""Exercício: a fila de julho com orçamento por dia da semana, e o jeito do Igor.

Rode de dentro da pasta desta aula:

    python3 orcamento.py

Complete as três funções marcadas com FAÇA. O resto do programa já monta a nota e imprime o
resultado. Depois rode `python3 testar_orcamento.py` para ver se as funções respeitam o pedido.
"""
import numpy as np
import pandas as pd

from fila import SORTEIO, preparar, carregar


def capacidade(dia):
    """Retorna (topo, sorteadas) para um dia no formato '2026-04-04'.

    FAÇA: use o pedido do Marcos em pedido-julho.md. Dica de pandas:
    pd.Timestamp(dia).dayofweek é 0 na segunda e 6 no domingo.
    """
    raise NotImplementedError('complete capacidade(dia)')


def fila_por_dia(novas, capacidade, sorteio):
    """Fila de cada dia com o orçamento daquele dia.

    FAÇA: para cada dia, as `topo` compras de nota mais alta (empate: menor id_transacao) e
    `sorteadas` compras sorteadas entre as que sobraram. Retorne as compras escolhidas com a
    coluna `origem` igual a 'topo' ou 'sorteada'. A função montar_fila de fila.py faz isso para
    um orçamento fixo e pode ser usada dia a dia.
    """
    raise NotImplementedError('complete fila_por_dia(novas, capacidade, sorteio)')


def fila_do_periodo(novas, total):
    """O jeito do Igor: as `total` compras de nota mais alta do período inteiro.

    FAÇA: ordene por nota (empate: menor id_transacao) e retorne as `total` primeiras.
    """
    raise NotImplementedError('complete fila_do_periodo(novas, total)')


def main():
    _, novas = preparar(carregar())
    sem_fraude = novas.drop(columns='fraude')
    fila = fila_por_dia(sem_fraude, capacidade, np.random.default_rng(SORTEIO))
    topo = fila[fila.origem == 'topo']
    igor = fila_do_periodo(sem_fraude, len(topo))
    fraude = novas['fraude']

    print('1. Fila por dia, com o orçamento de julho')
    print(f'   revisões: {len(fila)} ({len(topo)} do topo + {(fila.origem == "sorteada").sum()} sorteadas)')
    print(f'   fraudes: {int(fraude[topo.index].sum())} no topo, '
          f'{int(fraude[fila.index[fila.origem == "sorteada"]].sum())} nas sorteadas, de {int(fraude.sum())}')
    print('2. O jeito do Igor, com o mesmo número de revisões do topo')
    print(f'   revisões: {len(igor)}; fraudes: {int(fraude[igor.index].sum())}')
    # FAÇA: imprima em quantos dias o jeito do Igor passa do topo que a equipe tem naquele dia,
    # o maior número de compras dele num dia só e em quantos dias ele não pega compra nenhuma.


if __name__ == '__main__':
    main()
