"""Testes da montagem da fila. Rode de dentro da pasta desta aula:

    python3 testar_fila.py

Cada teste monta compras pequenas, com a resposta certa decidida antes, e olha a fila que sai.
"""
import unittest

import numpy as np
import pandas as pd

from fila import montar_fila


def compras(*dias):
    """dias: listas de notas, uma lista por dia. Os ids seguem a ordem."""
    linhas, i = [], 1
    for d, notas in enumerate(dias, start=1):
        for n in notas:
            linhas.append({'id_transacao': i, 'dia': f'2026-04-{d:02d}', 'nota': n, 'fraude': 0})
            i += 1
    return pd.DataFrame(linhas)


def sorteio():
    return np.random.default_rng(1)


class TestarFila(unittest.TestCase):
    def test_nunca_passa_do_orcamento_de_cada_dia(self):
        c = compras([9, 8, 7, 6, 5, 4, 3, 2], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        f = montar_fila(c, 5, 1, sorteio())
        self.assertEqual(f.groupby('dia').size().tolist(), [6, 6])

    def test_topo_sao_as_maiores_notas_do_dia_e_nao_do_periodo(self):
        # O dia 2 tem notas baixas, mas a equipe revisa todo dia: as 2 maiores do dia 2 entram.
        c = compras([9, 9, 9, 9], [1, 2, 3])
        f = montar_fila(c, 2, 0, sorteio())
        self.assertEqual(f[f.dia == '2026-04-02'].nota.tolist(), [3, 2])

    def test_sorteada_nunca_e_uma_do_topo(self):
        c = compras(list(range(20)))
        for semente in range(30):
            f = montar_fila(c, 5, 1, np.random.default_rng(semente))
            topo = set(f[f.origem == 'topo'].id_transacao)
            sorteada = set(f[f.origem == 'sorteada'].id_transacao)
            self.assertEqual(len(sorteada), 1)
            self.assertFalse(topo & sorteada)

    def test_dia_com_menos_compras_que_o_orcamento(self):
        c = compras([5, 3, 4])
        f = montar_fila(c, 5, 1, sorteio())
        self.assertEqual(f.id_transacao.tolist(), [1, 3, 2])
        self.assertEqual((f.origem == 'sorteada').sum(), 0)

    def test_empate_entra_a_compra_mais_antiga(self):
        c = compras([7, 7, 7])
        f = montar_fila(c, 2, 0, sorteio())
        self.assertEqual(f.id_transacao.tolist(), [1, 2])

    def test_a_coluna_fraude_nao_muda_a_fila(self):
        c = compras(list(range(12)), list(range(9)))
        trocada = c.assign(fraude=np.random.default_rng(3).integers(0, 2, len(c)))
        a = montar_fila(c, 3, 1, sorteio())
        b = montar_fila(trocada, 3, 1, sorteio())
        self.assertEqual(a.id_transacao.tolist(), b.id_transacao.tolist())

    def test_compra_sem_nota_para_a_fila(self):
        c = compras([3, 2, 1])
        c.loc[1, 'nota'] = np.nan
        with self.assertRaises(ValueError):
            montar_fila(c, 2, 1, sorteio())

    def test_orcamento_negativo_para_a_fila(self):
        with self.assertRaises(ValueError):
            montar_fila(compras([3, 2, 1]), -1, 1, sorteio())

    def test_coluna_de_ordem_que_nao_existe(self):
        with self.assertRaises(ValueError):
            montar_fila(compras([3, 2, 1]), 2, 1, sorteio(), por='notas')


if __name__ == '__main__':
    unittest.main(verbosity=2)
