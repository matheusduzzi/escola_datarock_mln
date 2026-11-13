# Clientes da Loja Exemplo

A Loja Exemplo não existe. Os clientes são números sorteados por `gerar.py`, que só usa a biblioteca padrão do Python. Nenhum dado vem de cliente, pessoa ou empresa real.

## Arquivo

`clientes.csv` tem 3000 linhas, uma por cliente que comprou pelo menos uma vez nos 12 meses até 2026-06-30. As colunas resumem as compras desse período. A base não tem coluna de grupo nem de resposta certa: os grupos são o que você vai procurar.

| Coluna | O que é |
|---|---|
| `id_cliente` | número do cliente, dado na ordem do cadastro |
| `data_cadastro` | dia em que o cliente criou a conta |
| `regiao` | região do endereço de entrega principal |
| `pedidos_12m` | quantos pedidos o cliente fez nos 12 meses |
| `gasto_12m` | quanto ele gastou nesses pedidos, em reais |
| `ticket_medio` | gasto dividido por pedidos, em reais |
| `dias_desde_ultima_compra` | dias entre a última compra e 2026-06-30 |
| `categorias_distintas` | em quantas categorias de produto ele comprou |
| `pct_pedidos_cupom` | porcentagem dos pedidos com cupom de desconto, de 0 a 100 |
| `pct_pedidos_app` | porcentagem dos pedidos feitos pelo app, de 0 a 100 |
| `devolucoes_12m` | quantos pedidos ele devolveu |
| `avaliacao_media` | média das notas de 1 a 5 que ele deu aos produtos; vazia quando ele nunca avaliou |

## Como gerar de novo

```bash
python3 gerar.py
```

A saída é `clientes=3000 pedidos=17541 avaliacao_vazia=945`. O arquivo sai igual toda vez. Nas aulas, você agrupa e avalia primeiro e só depois abre o gerador para comparar.

Não edite o CSV à mão.
