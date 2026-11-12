"""Gera os clientes da Loja Exemplo usados no curso.

Só biblioteca padrão (Python 3.9 ou mais novo). Semente fixa: rodar duas vezes dá o mesmo
arquivo, byte a byte.

    python3 gerar.py                 # escreve nesta pasta
    python3 gerar.py --saida /tmp/x  # escreve em outra pasta

Arquivo:
- clientes.csv: um cliente por linha, com o que ele comprou nos 12 meses até 2026-06-30.

Nas aulas, você agrupa e avalia primeiro e só depois lê este programa para comparar.
"""
import argparse
import csv
import datetime
import math
import pathlib
import random

SEED = 2712
DATA_CORTE = datetime.date(2026, 6, 30)
REGIOES = [("Sudeste", 45), ("Sul", 18), ("Centro-Oeste", 9), ("Nordeste", 20), ("Norte", 8)]

# Perfis de compra usados para sortear cada cliente. O perfil NÃO vai para o CSV.
# (nome, quantidade de clientes)
PERFIS = [("ocasional", 1350), ("frequente", 780), ("cupom", 480), ("alto_valor", 360), ("revenda", 30)]


def _sorteio(rng, opcoes):
    return rng.choices([o[0] for o in opcoes], weights=[o[1] for o in opcoes])[0]


def _poisson(rng, media):
    limite, k, p = math.exp(-media), 0, 1.0
    while True:
        p *= rng.random()
        if p <= limite:
            return k
        k += 1


def _entre(x, lo, hi):
    return max(lo, min(hi, x))


def _cliente(rng, perfil):
    if perfil == "ocasional":
        pedidos = 1 + _poisson(rng, 0.7)
        ticket = math.exp(rng.gauss(4.7, 0.55))
        recencia = rng.randint(40, 365)
        cupom = _entre(rng.gauss(0.12, 0.12), 0, 1)
        categorias = min(pedidos, 1 + _poisson(rng, 0.4))
        app = rng.random()
    elif perfil == "frequente":
        pedidos = int(_entre(round(rng.gauss(11, 3.5)), 5, 24))
        ticket = math.exp(rng.gauss(5.0, 0.35))
        recencia = rng.randint(1, 45)
        cupom = _entre(rng.gauss(0.15, 0.1), 0, 1)
        categorias = min(pedidos, 3 + _poisson(rng, 1.5))
        app = _entre(rng.gauss(0.78, 0.15), 0, 1)
    elif perfil == "cupom":
        pedidos = int(_entre(round(rng.gauss(6, 2.2)), 2, 14))
        ticket = math.exp(rng.gauss(4.2, 0.35))
        recencia = rng.randint(3, 120)
        cupom = _entre(rng.gauss(0.8, 0.12), 0, 1)
        categorias = min(pedidos, 2 + _poisson(rng, 1.2))
        app = _entre(rng.gauss(0.6, 0.2), 0, 1)
    elif perfil == "alto_valor":
        pedidos = 1 + _poisson(rng, 1.4)
        ticket = math.exp(rng.gauss(7.2, 0.4))
        recencia = rng.randint(5, 240)
        cupom = _entre(rng.gauss(0.05, 0.06), 0, 1)
        categorias = min(pedidos, 1 + _poisson(rng, 0.8))
        app = _entre(rng.gauss(0.35, 0.2), 0, 1)
    else:  # revenda
        pedidos = rng.randint(45, 130)
        ticket = math.exp(rng.gauss(6.0, 0.25))
        recencia = rng.randint(1, 10)
        cupom = _entre(rng.gauss(0.02, 0.02), 0, 1)
        categorias = 1 + _poisson(rng, 0.5)
        app = _entre(rng.gauss(0.05, 0.04), 0, 1)
    gasto = round(pedidos * ticket * rng.uniform(0.9, 1.1), 2)
    pedidos_cupom = int(round(cupom * pedidos))
    pedidos_app = int(round(app * pedidos))
    devolucoes = min(pedidos, _poisson(rng, 0.04 * pedidos + (0.3 if perfil == "alto_valor" else 0)))
    if rng.random() < (0.45 if perfil == "ocasional" else 0.2):
        avaliacao = ""
    else:
        avaliacao = f"{_entre(round(rng.gauss(4.2 if perfil != 'cupom' else 3.8, 0.6), 1), 1.0, 5.0):.1f}"
    anos = {"ocasional": 1.5, "frequente": 3.0, "cupom": 2.0, "alto_valor": 3.5, "revenda": 4.0}[perfil]
    cadastro = DATA_CORTE - datetime.timedelta(days=int(_entre(rng.expovariate(1 / (anos * 365)), recencia + 1, 2700)))
    return {
        "data_cadastro": cadastro.isoformat(),
        "regiao": _sorteio(rng, REGIOES),
        "pedidos_12m": pedidos,
        "gasto_12m": f"{gasto:.2f}",
        "ticket_medio": f"{gasto / pedidos:.2f}",
        "dias_desde_ultima_compra": recencia,
        "categorias_distintas": categorias,
        "pct_pedidos_cupom": round(100 * pedidos_cupom / pedidos),
        "pct_pedidos_app": round(100 * pedidos_app / pedidos),
        "devolucoes_12m": devolucoes,
        "avaliacao_media": avaliacao,
    }


def sortear():
    """Retorna a lista de (cliente, perfil) na ordem do id. O perfil não vai para o CSV."""
    rng = random.Random(SEED)
    perfis = [p for p, n in PERFIS for _ in range(n)]
    rng.shuffle(perfis)
    pares = [(_cliente(rng, p), p) for p in perfis]
    # O id segue a ordem do cadastro, como num sistema que numera o cliente quando ele se cadastra.
    pares.sort(key=lambda par: (par[0]["data_cadastro"], rng.random()))
    return pares


def gerar(saida):
    saida.mkdir(parents=True, exist_ok=True)
    clientes = [c for c, _ in sortear()]
    campos = ["id_cliente"] + list(clientes[0])
    with open(saida / "clientes.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=campos, lineterminator="\n")
        w.writeheader()
        for i, c in enumerate(clientes, start=1):
            w.writerow({"id_cliente": i, **c})
    vazios = sum(1 for c in clientes if c["avaliacao_media"] == "")
    return {"clientes": len(clientes), "pedidos": sum(c["pedidos_12m"] for c in clientes),
            "avaliacao_vazia": vazios}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gera clientes.csv da Loja Exemplo.")
    parser.add_argument("--saida", default=str(pathlib.Path(__file__).resolve().parent),
                        help="pasta onde o CSV é escrito")
    r = gerar(pathlib.Path(parser.parse_args().saida))
    print(f"clientes={r['clientes']} pedidos={r['pedidos']} avaliacao_vazia={r['avaliacao_vazia']}")
