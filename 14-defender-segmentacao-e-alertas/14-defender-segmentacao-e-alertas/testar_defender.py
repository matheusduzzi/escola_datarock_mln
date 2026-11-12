"""Testes das falhas que o relatório promete evitar. Rode de dentro da pasta desta aula:

    python3 testar_defender.py

Cada teste monta um caso pequeno com resposta conhecida. Passar nos testes mostra que o código
faz o que a aula diz; não mostra que os grupos servem para a campanha.
"""
import numpy as np
import pandas as pd

from alertas import carregar, com_nota, montar_fila, wilson
from segmentos import cabe, kmeans, preparar, sortear_sem_email, trocam

CLIENTES = pd.read_csv('../dados/clientes/clientes.csv')
COLUNAS = ['dias_desde_ultima_compra', 'pedidos_12m', 'gasto_12m', 'pct_pedidos_cupom']
testes = []


def teste(f):
    testes.append(f)
    return f


def recusa(f, trecho):
    try:
        f()
    except ValueError as e:
        assert trecho in str(e), f'mensagem diferente: {e}'
        return
    raise AssertionError('deveria recusar e passou')


@teste
def id_do_cliente_nao_entra():
    recusa(lambda: preparar(CLIENTES, COLUNAS + ['id_cliente']), 'ordem do cadastro')


@teste
def coluna_com_vazio_pede_decisao():
    recusa(lambda: preparar(CLIENTES, COLUNAS + ['avaliacao_media']), '945 vazios')
    marcado = CLIENTES.assign(avaliou=CLIENTES['avaliacao_media'].notna().astype(int))
    assert preparar(marcado, COLUNAS + ['avaliou']).shape == (3000, 5)


@teste
def texto_nao_vira_numero_sozinho():
    recusa(lambda: preparar(CLIENTES, COLUNAS + ['regiao']), 'não é número')


@teste
def cliente_extremo_nao_escapa_do_log():
    sem = preparar(CLIENTES, ['gasto_12m'])
    com = preparar(CLIENTES, ['gasto_12m'], log=['gasto_12m'])
    assert sem.max() > 8 and com.max() < 4, (sem.max(), com.max())


@teste
def mesma_partida_mesmos_grupos():
    Z = preparar(CLIENTES, COLUNAS, COLUNAS[:3])
    a, _ = kmeans(Z, 4, 7)
    b, _ = kmeans(Z, 4, 7)
    assert (a == b).all()


@teste
def trocar_o_numero_do_grupo_nao_conta_como_troca():
    a = np.array([0, 0, 1, 1, 2, 2])
    assert trocam(a, np.array([2, 2, 0, 0, 1, 1]), 3) == 0
    assert trocam(a, np.array([2, 2, 0, 1, 1, 1]), 3) == 1


@teste
def grupo_pequeno_nao_cabe():
    assert cabe([1290, 797, 491, 392], 4, 200)
    assert not cabe([1290, 797, 491, 392, 30], 5, 200)
    assert not cabe([1289, 825, 491, 395], 3, 200)


@teste
def todo_grupo_tem_quem_fica_sem_email():
    grupo = np.array([0] * 395 + [1] * 12)
    fora = sortear_sem_email(grupo, 0.1, 1)
    assert fora[grupo == 0].sum() == 40 and fora[grupo == 1].sum() == 1


@teste
def fila_nao_aceita_a_resposta():
    novas = com_nota(carregar())
    recusa(lambda: montar_fila(novas, 5, 1), 'fraude')


@teste
def fila_nunca_passa_do_orcamento_e_nao_olha_a_fraude():
    novas = com_nota(carregar())
    a = montar_fila(novas.drop(columns='fraude'), 5, 1)
    embaralhada = novas.assign(fraude=np.random.default_rng(3).permutation(novas['fraude'].to_numpy()))
    b = montar_fila(embaralhada.drop(columns='fraude'), 5, 1)
    assert a.groupby('dia').size().max() == 6 and list(a.index) == list(b.index)


@teste
def zero_no_sorteio_nao_e_zero_de_fraude():
    de, ate = wilson(0, 91)
    assert de == 0 and ate > 0.04
    recusa(lambda: wilson(1, 0), 'n > 0')


passaram = 0
for f in testes:
    try:
        f()
        passaram += 1
        print(f'ok     {f.__name__}')
    except Exception as e:  # noqa: BLE001
        print(f'FALHA  {f.__name__}: {type(e).__name__}: {e}')
print(f'{passaram} de {len(testes)} testes passaram')
raise SystemExit(0 if passaram == len(testes) else 1)
