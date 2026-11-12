# Compras com cartão da Loja Exemplo

A Loja Exemplo não existe. As compras são números sorteados por `gerar.py`, que só usa a biblioteca padrão do Python. Nenhum dado vem de cliente, cartão, pessoa ou empresa real.

## Arquivo

`transacoes.csv` tem 8000 linhas, em ordem de data e hora. Uma linha é uma compra paga com cartão no site ou no app entre 2026-01-01 e 2026-06-30.

| Coluna | O que é |
|---|---|
| `id_transacao` | número da compra, na ordem do tempo |
| `data_hora` | quando a compra foi feita |
| `canal` | `app` ou `site` |
| `valor` | valor da compra, em reais |
| `itens` | quantos itens a compra tem |
| `idade_conta_dias` | há quantos dias a conta do cliente existe |
| `tentativas_cartao` | quantas vezes o cartão foi digitado até a compra passar |
| `distancia_entrega_km` | distância entre o endereço de entrega e o endereço do cadastro |
| `hora` | hora do dia, de 0 a 23 |
| `fraude` | 1 quando o banco do cartão aceitou uma contestação de fraude; 0 nos outros casos |

Fraude é rara: são poucas dezenas de casos em 8000 compras. Antes de treinar, separe as partes pela data.

## Como gerar de novo

```bash
python3 gerar.py
```

A saída é `transacoes=8000 fraudes=101`. O arquivo sai igual toda vez. Nas aulas, você treina e avalia primeiro e só depois abre o gerador para comparar.

Não edite o CSV à mão.
