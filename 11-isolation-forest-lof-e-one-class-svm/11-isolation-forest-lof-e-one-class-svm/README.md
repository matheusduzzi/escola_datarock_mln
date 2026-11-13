# 11. Isolation Forest, LOF e One-Class SVM

Os arquivos da aula «Isolation Forest, LOF e One-Class SVM» do curso Aprendizado não supervisionado. Rode os comandos de dentro dessa pasta.

| Arquivo | Para que serve |
|---|---|
| `formas.py` | Um mapa de duas colunas com 252 pontos: uma mancha cheia, uma espalhada, um grupinho apertado e quatro pontos soltos (P1 a P4). Mostra em que posição cada método deixa cada um. |
| `compras.py` | Os três métodos nas compras com cartão da loja: aprendem com janeiro a março, dão nota a abril a junho e, no fim, contam quantas fraudes caíram nas 5 do topo de cada dia. |
| `mapa-de-adequacao.md` | O modelo do mapa que você preenche no exercício. |
| `requirements.txt` | As bibliotecas dessa aula e as versões em que ela foi rodada. |

As compras vêm de `../dados/fraude/transacoes.csv`, a base do curso.

## Como instalar

Com o Python 3.10 ou mais novo:

```bash
python3 -m pip install -r requirements.txt
```

Eu rodei com Python 3.14.5, scikit-learn 1.9.1, NumPy 2.5.4 e pandas 3.1.1. O scikit-learn 1.9.1 pede Python 3.11 ou mais novo. Com um Python mais antigo, o `pip` escolhe uma versão mais antiga.

## formas.py

```bash
python3 formas.py
python3 formas.py --vizinhos 5
python3 formas.py --nu 0.2 --gamma 2
python3 formas.py --pontos
```

| Opção | O que faz |
|---|---|
| `--vizinhos N` | Quantos vizinhos o LOF olha (`n_neighbors`). Sem ela, 20. |
| `--nu X` | O `nu` do One-Class SVM, maior que 0 e até 1. Sem ela, 0,05. |
| `--gamma X` | O `gamma` do One-Class SVM, um número maior que 0 ou `scale`. Sem ela, `scale`. |
| `--sorteio N` | Outro sorteio das árvores do Isolation Forest. Sem ela, 0. |
| `--pontos` | Lista os 252 pontos com as duas colunas e sai. |

## compras.py

```bash
python3 compras.py
python3 compras.py --vizinhos 50
python3 compras.py --sem-escala
python3 compras.py --referencia-sem-contestadas
```

`--vizinhos`, `--nu`, `--gamma` e `--sorteio` fazem o mesmo que no `formas.py`. As outras duas:

| Opção | O que faz |
|---|---|
| `--sem-escala` | Usa as colunas como estão, sem a escala z. |
| `--referencia-sem-contestadas` | Tira da referência de janeiro a março as compras que tiveram contestação de fraude. É o que dá para fazer quando essas contestações já chegaram. |

Os números aceitam vírgula ou ponto: `--nu 0,2` e `--nu 0.2` dão o mesmo resultado. Cada rodada leva alguns segundos.

## Se der erro

- `No module named 'sklearn'`: o scikit-learn não está no Python que você usou. Rode `python3 -m pip install -r requirements.txt` com o mesmo `python3`. No Windows, troque `python3` por `py`.
- `erro: não achei ../dados/fraude/transacoes.csv`: você rodou fora da pasta da aula. Entre em `11-isolation-forest-lof-e-one-class-svm/` e rode de novo.
- `error: argument` ou `--nu precisa ser`: um número de opção está fora do intervalo. A mensagem diz qual.
