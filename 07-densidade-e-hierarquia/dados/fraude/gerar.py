"""Gera as compras com cartão da Loja Exemplo usadas nas aulas de KNN.

Só biblioteca padrão (Python 3.9 ou mais novo). Semente fixa: rodar duas vezes dá o mesmo
arquivo, byte a byte.

    python3 gerar.py                 # escreve transacoes.csv nesta pasta
    python3 gerar.py --saida /tmp/x  # escreve em outra pasta

Uma linha é uma compra paga com cartão no site ou no app entre 2026-01-01 e 2026-06-30, com o
que o sistema sabe na hora de aprovar e o rótulo `fraude` (1 quando houve contestação de fraude
aceita pelo emissor, 0 caso contrário).

Nas aulas, você ajusta e avalia primeiro e só depois lê este programa para comparar.
"""
import argparse
import csv
import datetime
import math
import pathlib
import random

SEMENTE = 7331
N = 8000
INICIO = datetime.datetime(2026, 1, 1)
SEGUNDOS = int((datetime.datetime(2026, 7, 1) - INICIO).total_seconds())
TAXA_TIPO_A = 0.007  # conta nova, madrugada, entrega longe do cadastro
TAXA_TIPO_B = 0.005  # conta antiga, várias tentativas de cartão, valor alto


def _hora_normal(rng):
    return min(23, max(0, int(rng.gauss(15, 4))))


def gerar(saida):
    rng = random.Random(SEMENTE)
    saida.mkdir(parents=True, exist_ok=True)
    linhas = []
    for _ in range(N):
        quando = INICIO + datetime.timedelta(seconds=rng.randrange(SEGUNDOS))
        sorteio = rng.random()
        if sorteio < TAXA_TIPO_A:
            fraude, hora = 1, rng.choice([0, 1, 2, 3, 4, 5, 23])
            conta = rng.randint(0, 20)
            tentativas = rng.choice([1, 1, 2, 2, 3])
            distancia = round(rng.uniform(200, 1500), 1)
            valor = round(rng.uniform(300, 2500), 2)
        elif sorteio < TAXA_TIPO_A + TAXA_TIPO_B:
            fraude, hora = 1, _hora_normal(rng)
            conta = rng.randint(200, 2000)
            tentativas = rng.randint(4, 6)
            distancia = round(abs(rng.gauss(0, 30)), 1)
            valor = round(rng.uniform(1500, 5000), 2)
        else:
            fraude, hora = 0, _hora_normal(rng) if rng.random() > 0.05 else rng.randint(0, 5)
            conta = min(2000, int(rng.expovariate(1 / 500)))
            tentativas = 1 if rng.random() < 0.85 else rng.choice([2, 2, 3, 4])
            distancia = round(abs(rng.gauss(0, 25)), 1) if rng.random() < 0.93 else round(rng.uniform(50, 1200), 1)
            valor = round(math.exp(rng.gauss(math.log(180), 0.8)), 2)
        quando = quando.replace(hour=hora)
        itens = max(1, min(15, int(rng.expovariate(1 / 2.2)) + 1))
        canal = "app" if rng.random() < 0.55 else "site"
        linhas.append({
            "data_hora": quando.isoformat(sep=" "), "canal": canal,
            "valor": "%.2f" % valor, "itens": itens, "idade_conta_dias": conta,
            "tentativas_cartao": tentativas, "distancia_entrega_km": "%.1f" % distancia,
            "hora": hora, "fraude": fraude,
        })
    linhas.sort(key=lambda l: l["data_hora"])
    for i, l in enumerate(linhas, start=1):
        l["id_transacao"] = i
    colunas = ["id_transacao"] + [c for c in linhas[0] if c != "id_transacao"]
    with open(saida / "transacoes.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=colunas, lineterminator="\n")
        w.writeheader()
        w.writerows(linhas)
    return {"transacoes": len(linhas), "fraudes": sum(l["fraude"] for l in linhas)}


def main():
    parser = argparse.ArgumentParser(description="Gera transacoes.csv da Loja Exemplo.")
    parser.add_argument("--saida", default=str(pathlib.Path(__file__).resolve().parent),
                        help="pasta de saída (padrão: esta pasta)")
    resumo = gerar(pathlib.Path(parser.parse_args().saida))
    print(" ".join("%s=%s" % (k, v) for k, v in resumo.items()))


if __name__ == "__main__":
    main()
