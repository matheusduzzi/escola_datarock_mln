# 06. K-means e K-medoids

Os arquivos da aula «K-means e K-medoids» do curso Aprendizado não supervisionado. Rode os comandos de dentro dessa pasta.

| Arquivo | Para que serve |
|---|---|
| `a_mao.py` | K-means e K-medoids em poucos clientes, rodada por rodada, com a distância de cada cliente a cada centro. Serve para você fazer a conta à mão e depois ver se bateu. |
| `agrupar.py` | K-means e K-medoids nos 3000 clientes, com as oito colunas na escala z. Mostra o tamanho e o centro de cada grupo, a curva da soma por K e o que muda de um início para outro. |
| `minhas-respostas.md` | Onde você escreve as respostas da prática guiada e do exercício, com as contas feitas à mão. |

Os clientes vêm de `../dados/clientes/clientes.csv`, a base do curso.

## Como rodar

Os dois programas precisam do Python 3.9 ou mais novo com o NumPy e o pandas, como no README do kit do curso. O K-means e o K-medoids estão escritos com o NumPy, então não precisa de outra biblioteca.

### a_mao.py

```bash
python3 a_mao.py
python3 a_mao.py --inicio 1,105,143
python3 a_mao.py --metodo kmedoids --inicio 1,105,143
python3 a_mao.py --melhor 3
python3 a_mao.py --curva
```

Sem opção, ele mostra os 12 clientes da aula com duas colunas: `pedidos_12m` e `ticket_medio`. O ticket entra em centenas de reais, com uma casa: R$ 424,06 vira 4,2.

| Opção | O que faz |
|---|---|
| `--inicio id,id` | Roda o K-means a partir desses clientes, um centro por cliente, e mostra cada rodada: a distância de cada cliente a cada centro, o grupo e o centro novo. De 1 a 6 clientes. |
| `--metodo kmeans` ou `kmedoids` | Escolhe o método. Sem ela, é o K-means. No K-medoids, cada rodada mostra a soma das distâncias de cada cliente até os outros do grupo. |
| `--melhor K` | Roda a partir de todos os inícios possíveis e mostra o melhor resultado. No K-means, mostra também os outros lugares onde ele parou e quantos inícios foram para cada um. |
| `--curva` | O melhor resultado de cada método para K de 1 a 6, a queda da soma de um K para o outro e os grupos de um cliente só. |
| `--sem id,id` | Tira esses clientes da conta. |
| `--ids id,id` | Usa outros clientes no lugar dos 12 da aula, de 4 a 20. |
| `--escala centenas`, `reais` ou `z` | Como as colunas entram: ticket em centenas de reais (o padrão), ticket em reais, ou as duas colunas na escala z dos 3000 clientes. |
| `--distancia euclidiana` ou `manhattan` | A distância do K-medoids. O K-means usa sempre a euclidiana. |

As distâncias aparecem com uma casa. Se a sua conta à mão der 13,0 e a saída disser 13,0, bateu. Se der 13,1, veja se você arredondou no meio da conta.

### agrupar.py

```bash
python3 agrupar.py
python3 agrupar.py --metodo kmedoids
python3 agrupar.py --curva
python3 agrupar.py --repetir
python3 agrupar.py --sem-topo
```

| Opção | O que faz |
|---|---|
| `--metodo kmeans` ou `kmedoids` | Escolhe o método. Sem ela, é o K-means. |
| `--k K` | Quantos grupos, de 1 a 10. Sem ela, 3. |
| `--inicios N` | Quantos inícios sorteados, de 1 a 50. Fica o de menor soma. Sem ela, 10. |
| `--sorteio s` | O sorteio dos inícios. Sem ela, 1. Os sorteios saem iguais toda vez. |
| `--curva` | K-means com K de 1 a 10: a soma dos quadrados, a queda de um K para o outro e o menor grupo. |
| `--repetir` | 20 rodadas de um início só, com os sorteios de 1 a 20, para ver se todas chegam no mesmo lugar. |
| `--sem-topo` | Tira os 30 clientes de 40 ou mais pedidos antes de agrupar. |
| `--salvar arquivo.csv` | Grava `id_cliente` e `grupo`. |

A tabela de grupos mostra o tamanho de cada um, quantos clientes de 40 ou mais pedidos caíram nele e o centro em reais, dias e porcentagens. No K-means, o centro é a média do grupo. No K-medoids, é um cliente de verdade, e a última coluna diz qual.

O K-means sai na hora. O K-medoids leva 1 ou 2 segundos, porque monta a tabela de distâncias entre todos os pares de clientes.

## Se der erro

Se o programa parar com `erro:`, a mensagem diz o que mudar: cliente que não existe ou que não está na lista, cliente repetido, K fora da faixa, opção que só vale com outro método ou nome de arquivo que não termina em `.csv`.

Se aparecer `não achei ../dados/clientes/clientes.csv`, você está rodando de outra pasta. Entre na pasta dessa aula e rode de novo.
