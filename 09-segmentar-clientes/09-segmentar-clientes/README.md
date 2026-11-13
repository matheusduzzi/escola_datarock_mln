# 09. Segmentar clientes

Os arquivos da aula «Segmentar clientes» do curso Aprendizado não supervisionado. Rode os comandos de dentro dessa pasta.

| Arquivo | Para que serve |
|---|---|
| `segmentar.py` | Agrupa os 3000 clientes com as colunas, a escala e o método que você escolher, e mostra o perfil de cada grupo: tamanho, medianas, faixa do meio e o centro de cada grupo. Também grava o resultado e compara duas rodadas. |
| `requirements.txt` | As bibliotecas dessa aula e as versões em que ela foi rodada. |
| `minhas-respostas.md` | Onde você escreve as respostas da prática guiada e do exercício. |

Os clientes vêm de `../dados/clientes/clientes.csv`, a base do curso.

## Como instalar

A aula usa o scikit-learn e o SciPy, como as aulas «Densidade e hierarquia» e «Métricas internas». Se você já instalou para elas, não precisa instalar de novo. Com o Python 3.10 ou mais novo:

```bash
python3 -m pip install -r requirements.txt
```

Eu rodei com Python 3.14.5, scikit-learn 1.9.1, SciPy 1.18.1, NumPy 2.5.4 e pandas 3.1.1. O scikit-learn 1.9.1 pede Python 3.11 ou mais novo, e o SciPy 1.18.1 pede 3.12. Com um Python mais antigo, o `pip` escolhe versões mais antigas das duas bibliotecas.

## segmentar.py

```bash
python3 segmentar.py --colunas pedidos,gasto,ticket,dias,categorias,cupom,app,devolucoes --escala z --perfil
python3 segmentar.py --colunas pedidos,ticket,dias,categorias,cupom,app --escala log --topo separar --perfil
python3 segmentar.py --colunas pedidos,ticket,dias,categorias,cupom,app --escala log --topo separar --grupo 3
python3 segmentar.py --colunas pedidos,ticket,dias,categorias,cupom,app --escala z --topo separar --salvar z.csv
python3 segmentar.py --colunas pedidos,ticket,dias,categorias,cupom,app --escala log --topo separar --comparar z.csv
```

O `--colunas` e o `--escala` são obrigatórios: o programa não escolhe por você. As colunas têm nomes curtos:

| Nome | Coluna do arquivo |
|---|---|
| `pedidos` | `pedidos_12m` |
| `gasto` | `gasto_12m` |
| `ticket` | `ticket_medio` |
| `dias` | `dias_desde_ultima_compra` |
| `categorias` | `categorias_distintas` |
| `cupom` | `pct_pedidos_cupom` |
| `app` | `pct_pedidos_app` |
| `devolucoes` | `devolucoes_12m` |
| `avaliacao` | `avaliacao_media`, que pede o `--vazio` |
| `id` | `id_cliente` |

A região e a data de cadastro não entram na distância. Elas aparecem no `--perfil`.

| Opção | O que faz |
|---|---|
| `--colunas a,b,c` | As colunas que entram na distância, separadas por vírgula. Pelo menos duas. |
| `--escala z`, `log`, `robusta` ou `nenhuma` | `z`: cada coluna menos a média, dividida pelo desvio padrão. `log`: o log de 1 mais o valor nos pedidos, no gasto, no ticket e nas devoluções, e depois a `z` em todas. `robusta`: cada coluna menos a mediana, dividida pelo intervalo entre o primeiro e o terceiro quartil. `nenhuma`: as colunas como estão no arquivo. |
| `--topo separar` | Tira os clientes de 40 pedidos ou mais antes da escala e do agrupamento. Sem ela, eles ficam na conta (`--topo manter`). |
| `--vazio media` ou `zero` | Com a coluna `avaliacao`: o que pôr no lugar da avaliação de quem nunca avaliou. |
| `--metodo kmeans`, `ward` ou `hdbscan` | O método. Sem ela, `kmeans`. |
| `--k K` | Com `kmeans` ou `ward`: quantos grupos, de 2 a 10. Sem ela, 4. |
| `--min-grupo N` | Com `hdbscan`: o tamanho mínimo de um grupo, de 5 a 500. Sem ela, 50. Quem não cabe em grupo nenhum fica "sem grupo". |
| `--ver-escala` | Mostra como cada coluna da distância ficou na escala escolhida, do menor ao maior cliente. |
| `--perfil` | O perfil de cada grupo: tamanho, medianas, porcentagens, região mais comum e o centro do grupo nas colunas da distância. |
| `--grupo N` | A faixa do meio (25% a 75%) e o maior valor de cada coluna num grupo, e quantos clientes dele usam cupom em metade dos pedidos ou mais. |
| `--salvar arquivo.csv` | Grava `id_cliente` e `grupo` de cada cliente. Quem saiu pelo `--topo separar` sai como `topo`; quem ficou sem grupo, como `-1`. |
| `--comparar arquivo.csv` | Compara a rodada de agora com uma gravada pelo `--salvar`: a tabela cruzada dos grupos, o casamento entre eles e quantos clientes trocam de grupo. |
| `--gerador` | Cruza os grupos com o jeito de comprar que o `gerar.py` sorteou para cada cliente. Use só depois de escolher a sua segmentação. |

Os grupos saem numerados do maior para o menor. A numeração pode mudar de uma rodada para outra; o `--comparar` casa os grupos antes de contar quem trocou.

A silhouette sai "nessa representação": com as colunas, a escala e os clientes da rodada. Ela só compara rodadas iguais nesses três pontos. O `--metodo ward` leva alguns segundos, porque monta a árvore com todos os pares de clientes, e a silhouette também.

## Se der erro

Se o programa parar com `erro:`, a mensagem diz o que mudar: coluna que não existe ou repetida, região ou data na distância, avaliação sem `--vazio`, K, grupo mínimo ou número do grupo fora da faixa, opção que só vale com outra, coluna com intervalo entre quartis 0 na escala robusta, arquivo que não termina em `.csv` ou que não veio do `--salvar`.

Se aparecer `não achei ../dados/clientes/clientes.csv`, você está rodando de outra pasta. Entre na pasta dessa aula e rode de novo.

Se aparecer `ModuleNotFoundError: No module named 'sklearn'` ou `'scipy'`, faltou instalar as bibliotecas. Rode o `pip install` do começo desse arquivo.
