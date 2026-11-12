# 04. Kernel PCA, t-SNE e UMAP

Os arquivos da aula «Kernel PCA, t-SNE e UMAP» do curso Aprendizado não supervisionado. Rode os comandos de dentro dessa pasta.

| Arquivo | Para que serve |
|---|---|
| `aneis.py` | Dois anéis, um dentro do outro. Mostra o que o PCA e o Kernel PCA fazem com eles, com sete valores de gamma, e o tamanho da tabela de semelhanças. |
| `projecoes.py` | PCA, Kernel PCA e t-SNE nos clientes da loja, com três medidas: vizinhos mantidos, correlação das distâncias e afastamento dos maiores clientes. |
| `minhas-respostas.md` | Onde você escreve as respostas da prática e a comparação do exercício. |

Os clientes vêm de `../dados/clientes/clientes.csv`, a base do curso. O `projecoes.py` sorteia 1.000 dos 3.000 clientes, sempre os mesmos, para o t-SNE rodar em segundos.

## Como rodar

Você precisa do Python 3.9 ou mais novo com o NumPy e o pandas, como no README do kit do curso. Essa aula não usa outra biblioteca: o Kernel PCA e o t-SNE estão escritos com o NumPy, dentro do `projecoes.py`. O UMAP entra na aula 05, com a versão e o comando de instalação.

```bash
python3 aneis.py
python3 projecoes.py --comparar
python3 projecoes.py --metodo kpca
python3 projecoes.py --metodo tsne --perplexidade 5
```

| Opção | O que faz |
|---|---|
| `--metodo pca`, `kpca` ou `tsne` | Escolhe o método. Sem ela, é o PCA. |
| `--comparar` | Roda os três métodos com os valores padrão e mostra as medidas lado a lado. |
| `--gamma g` | Gamma do Kernel PCA. Sem ela, 1 dividido pelo número de colunas, como no scikit-learn. Aceita vírgula ou ponto: `0,01` ou `0.01`. |
| `--perplexidade p` | Perplexidade do t-SNE. Sem ela, 30. Precisa ficar entre 2 e um terço do número de clientes. |
| `--inicio aleatorio` e `--sorteio s` | O t-SNE começa de pontos sorteados, não do PCA. |
| `--repetir` | Roda o t-SNE duas vezes, com os inícios sorteados 1 e 2, e compara um desenho com o outro. |
| `--mais N` | Chegam N clientes novos, de fora do sorteio. Mostra o que cada método faz com eles. |
| `--clientes N` | Sorteia N clientes, de 100 a 3.000, em vez de 1.000. |
| `--salvar arquivo.csv` | Grava `id_cliente`, `x`, `y` e `grande` (1 para quem fez 40 ou mais pedidos) para você desenhar na ferramenta que quiser. |

O PCA e o Kernel PCA saem na hora. O t-SNE leva alguns segundos com 1.000 clientes. Com `--mais 500`, uns 20 segundos, e com `--mais 2000`, perto de um minuto: o t-SNE roda de novo com todos, e a conta cresce com o quadrado do número de clientes. Os sorteios saem iguais toda vez.

Se o programa parar com `erro:`, a mensagem diz o que mudar: opção que não vale para o método escolhido, gamma ou perplexidade fora da faixa, `--mais` maior que os clientes de fora do sorteio ou arquivo que não termina em `.csv`.

Se aparecer `não achei ../dados/clientes/clientes.csv`, você está rodando de outra pasta. Entre na pasta dessa aula e rode de novo.
