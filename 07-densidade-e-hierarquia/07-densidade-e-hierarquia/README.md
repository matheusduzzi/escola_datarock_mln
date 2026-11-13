# 07. Densidade e hierarquia

Os arquivos da aula «Densidade e hierarquia» do curso Aprendizado não supervisionado. Rode os comandos de dentro dessa pasta.

| Arquivo | Para que serve |
|---|---|
| `a_mao.py` | DBSCAN, HDBSCAN e a árvore do agrupamento hierárquico em poucos clientes, passo a passo: vizinhos no raio, núcleo, borda e ruído, distância de núcleo e as junções da árvore com a altura de cada uma. Serve para você fazer a conta à mão e depois ver se bateu. |
| `formas.py` | Três formas de grupo em duas dimensões (dois anéis, densidades diferentes, duas manchas e uma ponte) e o que o K-means, o Ward, a menor distância, o DBSCAN e o HDBSCAN fazem com cada uma. |
| `base.py` | DBSCAN, HDBSCAN e a árvore nos 3000 clientes, com as oito colunas na escala z. Mostra o tamanho e a média de cada grupo e do ruído. |
| `requirements.txt` | As bibliotecas dessa aula e as versões em que ela foi rodada. |
| `minhas-respostas.md` | Onde você escreve as respostas da prática guiada e do exercício, com as contas feitas à mão. |

Os clientes vêm de `../dados/clientes/clientes.csv`, a base do curso.

## Como instalar

Essa é a primeira aula do curso que usa o scikit-learn e o SciPy. Com o Python 3.10 ou mais novo:

```bash
python3 -m pip install -r requirements.txt
```

Eu rodei com Python 3.14.5, scikit-learn 1.9.1, SciPy 1.18.1, NumPy 2.5.4 e pandas 3.1.1. O scikit-learn 1.9.1 pede Python 3.11 ou mais novo, e o SciPy 1.18.1 pede 3.12. Com um Python mais antigo, o `pip` escolhe versões mais antigas das duas bibliotecas. O HDBSCAN está no scikit-learn desde a versão 1.3.

## a_mao.py

```bash
python3 a_mao.py
python3 a_mao.py --dbscan 3 --min 3
python3 a_mao.py --varrer --min 3
python3 a_mao.py --hdbscan 3
python3 a_mao.py --hierarquia single --altura 5
python3 a_mao.py --hierarquia ward --cortar 4
```

Sem opção, ele mostra os 12 clientes da aula com duas colunas, `pedidos_12m` e `ticket_medio`, e a distância entre cada par. O ticket entra em centenas de reais, com uma casa: R$ 424,06 vira 4,2.

| Opção | O que faz |
|---|---|
| `--dbscan RAIO` | Roda o DBSCAN com esse raio. Mostra os vizinhos de cada cliente no raio, quantos são contando ele mesmo, se ele é núcleo, borda ou ruído, e o grupo. No fim, diz se o `DBSCAN` do scikit-learn deu os mesmos grupos. |
| `--varrer` | Roda o DBSCAN com raios de 0,5 a 35 e mostra os grupos e o ruído de cada um. |
| `--min N` | O mínimo de clientes no raio para ser núcleo, contando o próprio cliente (`min_samples`). Vale com `--dbscan`, `--varrer` e `--hdbscan`. No DBSCAN, sem ela, é 3. No HDBSCAN, sem ela, é o mesmo número do `--hdbscan`. |
| `--hdbscan N` | Roda o HDBSCAN com tamanho mínimo de grupo N (`min_cluster_size`). Mostra a distância de núcleo de cada cliente, as junções pela distância de alcance, os grupos e a força de cada cliente no grupo. |
| `--hierarquia single`, `complete`, `average` ou `ward` | Mostra as junções da árvore, da menor altura para a maior, e os três maiores saltos de altura. |
| `--cortar K` ou `--altura H` | Com `--hierarquia`: corta a árvore em K grupos ou na altura H e mostra os grupos. |
| `--ids id,id` | Usa outros clientes no lugar dos 12 da aula, de 4 a 20. |
| `--sem id,id` | Tira esses clientes da conta. |

O raio e a altura aceitam vírgula ou ponto: `--dbscan 2,5` e `--dbscan 2.5` dão o mesmo resultado.

As distâncias da tabela aparecem com uma casa, e as alturas da árvore com duas. Se a sua conta à mão der 2,0 e a saída disser 2,00, bateu.

## formas.py

```bash
python3 formas.py
python3 formas.py --forma aneis
python3 formas.py --forma densidades --varrer
python3 formas.py --forma ponte --raio 0.8
```

| Opção | O que faz |
|---|---|
| `--forma aneis`, `densidades`, `ponte` ou `todas` | Escolhe a forma. Sem ela, as três. |
| `--raio R` | O raio do DBSCAN. Sem ela, 1,0 nos anéis, 0,7 nas densidades e 0,5 na ponte. O mínimo é sempre 5. |
| `--tamanho N` | O tamanho mínimo de grupo do HDBSCAN. Sem ela, 5 nos anéis e nas densidades e 10 na ponte. |
| `--varrer` | Roda o DBSCAN com raios de 0,2 a 1,2. |

Nas formas, cada ponto tem uma origem: a mancha ou o anel de onde ele saiu, ou nenhuma, para os pontos soltos e os da ponte. "Certos" conta os pontos que caíram no grupo da maioria da sua origem. "Errados" conta os outros. "Ruído (de grupo)" conta os pontos com origem que foram para o ruído. A última coluna conta quantos pontos sem origem foram para o ruído, que é onde eles deviam estar.

## base.py

```bash
python3 base.py --vizinho 8
python3 base.py --varrer --min 8
python3 base.py --dbscan 2.5 --min 8
python3 base.py --hdbscan 50
python3 base.py --hierarquia ward
python3 base.py --hierarquia ward --k 5
```

| Opção | O que faz |
|---|---|
| `--vizinho N` | Para cada cliente, a distância até o N-ésimo cliente mais perto, contando ele mesmo. Mostra os quantis e quantos clientes ficam acima de cada raio. Ajuda a escolher o raio do DBSCAN. |
| `--varrer` | DBSCAN com raios de 0,5 a 3,0: grupos, ruído, maior grupo e quantos clientes do topo caíram no ruído. |
| `--dbscan RAIO` | DBSCAN com esse raio. Sem `--min`, o mínimo é 8. |
| `--hdbscan N` | HDBSCAN com tamanho mínimo de grupo N. Sem `--min`, o `min_samples` é o mesmo N. |
| `--hierarquia single`, `complete`, `average` ou `ward` | Sem `--k`, mostra os tamanhos dos grupos quando a árvore é cortada em 1 a 10 grupos. Com `--k`, a tabela de um corte. |
| `--k K` | Com `--hierarquia`: em quantos grupos cortar, de 2 a 10. |
| `--ruido-por-ward K` | Com `--dbscan` ou `--hdbscan`: quantos clientes de cada grupo do Ward com K grupos foram para o ruído. |
| `--sem-topo` | Tira os 30 clientes de 40 ou mais pedidos antes de agrupar. |
| `--salvar arquivo.csv` | Grava `id_cliente` e `grupo`. O ruído sai como -1. |

A tabela de grupos mostra o tamanho de cada um, quantos clientes de 40 ou mais pedidos caíram nele e a média das oito colunas em reais, dias e porcentagens. A última linha é o ruído.

O `--hierarquia` e o `--ruido-por-ward` levam alguns segundos, porque a árvore usa a distância entre todos os pares de clientes.

## Se der erro

Se o programa parar com `erro:`, a mensagem diz o que mudar: cliente que não existe ou que não está na lista, cliente repetido, raio, mínimo ou K fora da faixa, opção que só vale com outra, ou nome de arquivo que não termina em `.csv`.

Se aparecer `não achei ../dados/clientes/clientes.csv`, você está rodando de outra pasta. Entre na pasta dessa aula e rode de novo.

Se aparecer `ModuleNotFoundError: No module named 'sklearn'` ou `'scipy'`, faltou instalar as bibliotecas. Rode o `pip install` do começo desse arquivo.
