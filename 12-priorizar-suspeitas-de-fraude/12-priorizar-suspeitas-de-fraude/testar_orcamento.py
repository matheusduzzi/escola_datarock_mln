"""Testes do exercício. Rode de dentro da pasta desta aula, depois de completar orcamento.py:

    python3 testar_orcamento.py

Eles olham se as funções seguem o pedido do Marcos. Não dizem quantas fraudes a fila pega.
"""
import unittest

import numpy as np
import pandas as pd

from orcamento import capacidade, fila_do_periodo, fila_por_dia


def compras(notas_por_dia):
    linhas, i = [], 1
    for dia, notas in notas_por_dia.items():
        for n in notas:
            linhas.append({'id_transacao': i, 'dia': dia, 'nota': n})
            i += 1
    return pd.DataFrame(linhas)


class TestarOrcamento(unittest.TestCase):
    def test_capacidade_segue_o_pedido(self):
        self.assertEqual(tuple(capacidade('2026-04-03')), (5, 1))  # sexta
        self.assertEqual(tuple(capacidade('2026-04-04')), (2, 1))  # sábado
        self.assertEqual(tuple(capacidade('2026-04-05')), (2, 1))  # domingo
        self.assertEqual(tuple(capacidade('2026-04-06')), (5, 1))  # segunda

    def test_fila_por_dia_respeita_o_orcamento_de_cada_dia(self):
        c = compras({'2026-04-03': list(range(10)), '2026-04-04': list(range(10, 20))})
        f = fila_por_dia(c, capacidade, np.random.default_rng(1))
        sexta, sabado = f[f.dia == '2026-04-03'], f[f.dia == '2026-04-04']
        self.assertEqual(sorted(sexta[sexta.origem == 'topo'].nota), [5, 6, 7, 8, 9])
        self.assertEqual(sorted(sabado[sabado.origem == 'topo'].nota), [18, 19])
        self.assertEqual((sexta.origem == 'sorteada').sum(), 1)
        self.assertEqual((sabado.origem == 'sorteada').sum(), 1)
        self.assertFalse(set(f[f.origem == 'sorteada'].id_transacao) & set(f[f.origem == 'topo'].id_transacao))

    def test_fila_do_periodo_pega_as_maiores_sem_olhar_o_dia(self):
        c = compras({'2026-04-03': [9, 8, 7], '2026-04-04': [1, 2, 3]})
        f = fila_do_periodo(c, 3)
        self.assertEqual(f.nota.tolist(), [9, 8, 7])

    def test_fila_do_periodo_desempata_pela_compra_mais_antiga(self):
        c = compras({'2026-04-03': [5, 5], '2026-04-04': [5]})
        self.assertEqual(fila_do_periodo(c, 2).id_transacao.tolist(), [1, 2])


if __name__ == '__main__':
    unittest.main(verbosity=2)
