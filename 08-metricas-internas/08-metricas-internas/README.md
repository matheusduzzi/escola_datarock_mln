# 08. Métricas internas

Os arquivos da aula «Métricas internas» do curso Aprendizado não supervisionado. Rode os comandos de dentro dessa pasta.

| Arquivo | Para que serve |
|---|---|
| `a_mao.py` | A silhouette, o Davies-Bouldin e o Calinski-Harabasz em poucos clientes, com todas as contas: o a, o b e o s de cada cliente, o centro e o espalhamento de cada grupo, e o espalhamento entre e dentro dos grupos. Serve para você fazer a conta à mão e depois ver se bateu. |
| `formas.py` | As três notas em duas formas que dá para desenhar: dois anéis e três manchas redondas. Mostra o que cada nota dá para a resposta certa e para o K-means. |
| `base.py` | As três notas nos 3000 clientes: as propostas da aula, cada método em cada representação, a nota de cada grupo e a curva de 2 a 10 grupos. |
| `requirements.txt` | As bibliotecas dessa aula e as versões em que ela foi rodada. |
| `minhas-respostas.md` | Onde você escreve as respostas da prática guiada e do exercício, com as contas feitas à mão. |

Os clientes vêm de `../dados/clientes/clientes.csv`, a base do curso.

## Como instalar

A aula usa o scikit-learn e o SciPy, como a aula «Densidade e hierarquia». Se você já instalou para ela, não precisa instalar de novo. Com o Python 3.10 ou mais novo:

```bash
python3 -m pip install -r requirements.txt
```

Eu rodei com Python 3.14.5, scikit-learn 1.9.1, SciPy 1.18.1, NumPy 2.5.4 e pandas 3.1.1. O scikit-learn 1.9.1 pede Python 3.11 ou mais novo, e o SciPy 1.18.1 pede 3.12. Com um Python mais antigo, o `pip` escolhe versões mais antigas das duas bibliotecas.

## a_mao.py

```bash
python3 a_mao.py
python3 a_mao.py --grupos 27,78,104,1/15,23,105,137,77/63,98,143
python3 a_mao.py --grupos 27,78,104,1/15,23,105,137,77/63,98,143 --cliente 105
python3 a_mao.py --grupos 27,78,104,1/15,23,105/63,98,143 --ruido 137,77
```

Sem opção, ele mostra os 12 clientes da aula com duas colunas, `pedidos_12m` e `ticket_medio`, e a distância entre cada par. O ticket entra em centenas de reais, com uma casa: R$ 424,06 vira 4,2.

| Opção | O que faz |
|---|---|
| `--grupos` | Os grupos que você quer medir. Os clientes de um grupo vão separados por vírgula, e os grupos por barra. Todo cliente da conta precisa estar num grupo ou no `--ruido`. Ele mostra as três notas com as contas e, no fim, diz se as funções do scikit-learn deram os mesmos números. |
| `--cliente ID` | Com `--grupos`: a conta da silhouette desse cliente, distância por distância. |
| `--ruido id,id` | Com `--grupos`: os clientes que ficaram fora de grupo, como o ruído do DBSCAN. As notas saem sem eles, e depois com eles como se fossem mais um grupo. |
| `--escala centenas`, `reais` ou `z` | Como o ticket entra na conta. Sem ela, centenas de reais. Em `z`, as duas colunas entram na escala z dos 3000 clientes. |
| `--ids id,id` | Usa outros clientes no lugar dos 12 da aula, de 4 a 20. |
| `--sem id,id` | Tira esses clientes da conta. |

As distâncias da tabela aparecem com uma casa, e o a e o b com duas. O programa faz a conta sem arredondar no meio. Se a sua conta à mão der 0,27 e a saída disser 0,28, olhe se você arredondou as distâncias antes de somar.

## formas.py

```bash
python3 formas.py
python3 formas.py --forma aneis
```

Nas formas, cada ponto tem uma origem: o anel ou a mancha de onde ele saiu. A linha "a origem dos pontos" mede essa resposta certa. "Errados" conta os pontos que caíram fora do grupo da maioria da sua origem.

## base.py

```bash
python3 base.py --propostas
python3 base.py --propostas --medido-em z
python3 base.py --grupos ward --k 5 --por-grupo
python3 base.py --grupos kmeans --feito-em reais --k 3 --medido-em todas
python3 base.py --grupos dbscan --medido-em z
python3 base.py --curva ward
```

O programa separa duas coisas: onde os grupos são feitos e onde as notas são medidas. As duas usam os mesmos nomes:

| Nome | O que é |
|---|---|
| `z` | as oito colunas de comportamento na escala z, como na aula «PCA» |
| `pca2` | os dois primeiros componentes do PCA dessas oito colunas na escala z |
| `reais` | as oito colunas como estão no arquivo, em reais, dias, pedidos e porcentagens |

| Opção | O que faz |
|---|---|
| `--propostas` | As três propostas da aula: a da agência (K-means com K = 3 nas colunas em reais), a do BI (K-means com K = 3 nos dois componentes do PCA) e a de Ward com 5 grupos na escala z. |
| `--grupos kmeans`, `ward` ou `dbscan` | Faz os grupos com esse método. |
| `--feito-em z`, `pca2` ou `reais` | Com `--grupos`: onde os grupos são feitos. Sem ela, `z`. |
| `--medido-em z`, `pca2`, `reais` ou `todas` | Onde as notas são medidas. Sem ela, no mesmo lugar onde os grupos foram feitos. `todas` mostra as três linhas e só vale com `--grupos`. |
| `--k K` | Com `kmeans` ou `ward`: quantos grupos, de 2 a 10. Sem ela, 5. |
| `--raio R` e `--min N` | Com `dbscan`: o raio e o mínimo de clientes no raio. Sem elas, 2,5 e 8, como na aula «Densidade e hierarquia». |
| `--por-grupo` | Com `--grupos`: a silhouette média de cada grupo, quantos clientes dele ficaram abaixo de 0 e a média das oito colunas. |
| `--curva kmeans` ou `ward` | As três notas para 2 a 10 grupos, com os grupos feitos na escala z. Com `--medido-em`, as notas saem medidas em outro lugar. |
| `--sem-topo` | Tira os 30 clientes de 40 ou mais pedidos antes de tudo. |
| `--salvar arquivo.csv` | Com `--grupos`: grava `id_cliente`, `grupo` e a silhouette de cada cliente. O ruído sai como -1. |

Os grupos saem numerados do maior para o menor. O `--grupos ward` e o `--curva ward` levam alguns segundos, porque a árvore usa a distância entre todos os pares de clientes. A silhouette também: ela precisa dessas mesmas distâncias.

## Se der erro

Se o programa parar com `erro:`, a mensagem diz o que mudar: cliente que não existe, repetido ou fora de grupo, grupo vazio, K, raio ou mínimo fora da faixa, opção que só vale com outra, ou nome de arquivo que não termina em `.csv`.

Se aparecer `não achei ../dados/clientes/clientes.csv`, você está rodando de outra pasta. Entre na pasta dessa aula e rode de novo.

Se aparecer `ModuleNotFoundError: No module named 'sklearn'` ou `'scipy'`, faltou instalar as bibliotecas. Rode o `pip install` do começo desse arquivo.
