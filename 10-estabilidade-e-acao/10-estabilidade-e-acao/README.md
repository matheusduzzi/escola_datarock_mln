# 10. Estabilidade e ação

Os arquivos da aula «Estabilidade e ação» do curso Aprendizado não supervisionado. Rode os comandos de dentro dessa pasta.

| Arquivo | Para que serve |
|---|---|
| `estabilidade.py` | Refaz a segmentação de agosto com outros pontos de partida, com sorteios de parte dos clientes e com outras escolhas, e conta quantos clientes trocam de grupo. Também divide cada grupo por sorteio para o teste da campanha. |
| `plano-de-uso.md` | O modelo do plano de uso dos grupos, para preencher no exercício. |
| `requirements.txt` | As bibliotecas dessa aula e as versões em que ela foi rodada. |
| `minhas-respostas.md` | Onde você escreve as respostas da prática guiada e do exercício. |

Os clientes vêm de `../dados/clientes/clientes.csv`, a base do curso.

## Como instalar

A aula usa o scikit-learn e o SciPy, como as aulas «Densidade e hierarquia», «Métricas internas» e «Segmentar clientes». Se você já instalou para elas, não precisa instalar de novo. Com o Python 3.10 ou mais novo:

```bash
python3 -m pip install -r requirements.txt
```

Eu rodei com Python 3.14.5, scikit-learn 1.9.1, SciPy 1.18.1, NumPy 2.5.4 e pandas 3.1.1. O scikit-learn 1.9.1 pede Python 3.11 ou mais novo, e o SciPy 1.18.1 pede 3.12. Com um Python mais antigo, o `pip` escolhe versões mais antigas das duas bibliotecas.

## estabilidade.py

O programa parte da segmentação de agosto da aula «Segmentar clientes»: as colunas `pedidos,ticket,dias,categorias,cupom,app`, log nos pedidos e no ticket e depois a escala z, os clientes de 40 pedidos ou mais separados antes, K-means com K = 4 e 10 começos. Cada opção muda uma dessas escolhas e deixa as outras como estão.

```bash
python3 estabilidade.py partida
python3 estabilidade.py partida --comecos 1
python3 estabilidade.py amostra
python3 estabilidade.py amostra --ver 18
python3 estabilidade.py amostra --comecos 50 --fronteira
python3 estabilidade.py comparar --k 3
python3 estabilidade.py teste
```

| Modo | O que faz |
|---|---|
| `partida` | Roda o mesmo K-means com outros pontos de partida (de 1 a 19; a 0 é a de agosto) e conta quantos clientes trocam de grupo em cada um. Mostra também a soma das distâncias ao quadrado de cada cliente ao centro do grupo: o K-means fica com o arranjo de menor soma. |
| `amostra` | Sorteia parte dos clientes, refaz a escala e o K-means só com eles e compara com agosto, nos clientes sorteados. Mostra quantos trocam, o ARI de cada sorteio, o Jaccard de cada grupo e quantos clientes ficaram no grupo em todos os sorteios. |
| `comparar` | Refaz a segmentação com a escolha que você mudou (`--k`, `--escala`, `--topo`, `--colunas` ou `--comecos`) e compara com a de agosto, nos 3000 clientes. |
| `teste` | Divide cada grupo de agosto por sorteio: quem recebe o e-mail do grupo, quem recebe o e-mail padrão e quem fica sem e-mail. Mostra a margem de cada comparação. |

| Opção | Vale em | O que faz |
|---|---|---|
| `--k K` | todos | Quantos grupos, de 2 a 8. Sem ela, 4. |
| `--escala z` ou `log` | todos | `log`: o log de 1 mais o valor nos pedidos, no gasto, no ticket e nas devoluções, e depois a `z`. Sem ela, `log`. |
| `--topo N` | todos | Separa antes de agrupar quem tem N pedidos ou mais. `--topo 0` não separa ninguém. Sem ela, 40. |
| `--colunas a,b,c` | todos | As colunas da distância. Os nomes são os da aula «Segmentar clientes»: `pedidos`, `gasto`, `ticket`, `dias`, `categorias`, `cupom`, `app`, `devolucoes`. |
| `--comecos N` | todos | Quantos começos o K-means testa em cada rodada, de 1 a 100. Ele fica com o de menor soma das distâncias. Sem ela, 10. |
| `--vezes N` | `partida`, `amostra` | Quantas partidas ou sorteios, de 2 a 50. Sem ela, 20. |
| `--fracao P` | `amostra` | A porcentagem de clientes em cada sorteio, de 50 a 95. Sem ela, 80. |
| `--ver N` | `amostra` | Abre o sorteio N: a tabela cruzada com agosto, as medianas dos grupos do sorteio e a soma das distâncias com 10 e com 50 começos (e com os começos pedidos, se forem outros). |
| `--fronteira` | `amostra` | Mostra quem ficou no grupo de agosto em menos de 90% dos sorteios em que entrou: de que grupo são e a mediana deles. |
| `--salvar arquivo.csv` | `amostra`, `teste` | No `amostra`, grava `id_cliente`, `grupo` e `firmeza` (a fração dos sorteios em que o cliente ficou no grupo). No `teste`, grava `id_cliente`, `grupo` e `email` (o e-mail que cada cliente recebe). |
| `--sem-email P`, `--padrao P` | `teste` | A porcentagem de cada grupo que fica sem e-mail e a que recebe o e-mail padrão, de 5 a 30. Sem elas, 10 e 10. |
| `--taxa P` | `teste` | A porcentagem de clientes que você imagina que voltam a comprar, de 1 a 99. Ela só entra na conta da margem. Sem ela, 20. |

Os sorteios saem iguais toda vez: o sorteio 18 de hoje é o sorteio 18 de amanhã. Os grupos saem numerados do maior para o menor, e a numeração pode mudar de uma rodada para outra; o programa casa os grupos antes de contar quem trocou.

O `amostra` com 20 sorteios leva alguns segundos. Com `--comecos 50`, leva mais.

## Se der erro

Se o programa parar com `erro:`, a mensagem diz o que mudar: coluna que não existe ou repetida, K, começos, vezes, fração, sorteio do `--ver` ou porcentagem do `teste` fora da faixa, ou arquivo do `--salvar` que não termina em `.csv`.

Se aparecer `não achei ../dados/clientes/clientes.csv`, você está rodando de outra pasta. Entre na pasta dessa aula e rode de novo.

Se aparecer `ModuleNotFoundError: No module named 'sklearn'` ou `'scipy'`, faltou instalar as bibliotecas. Rode o `pip install` do começo desse arquivo.
