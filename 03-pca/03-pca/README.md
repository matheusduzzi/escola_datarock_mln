# 03. PCA

Os arquivos da aula «PCA» do curso Aprendizado não supervisionado. Rode os comandos de dentro dessa pasta.

| Arquivo | Para que serve |
|---|---|
| `quatro_clientes.py` | O PCA de quatro clientes, passo a passo: centralizar, covariância, direções, reconstrução com um componente e o mesmo sem centralizar. Use para ver se a sua conta à mão bateu. |
| `pca_clientes.py` | O PCA nas colunas dos clientes: variância de cada componente, peso de cada coluna, reconstrução com k componentes e estabilidade em duas metades. |
| `minhas-respostas.md` | Onde você escreve as respostas da prática e a escolha de componentes do exercício. |

Os clientes vêm de `../dados/clientes/clientes.csv`, a base do curso.

## Como rodar

Você precisa do Python 3.9 ou mais novo com o NumPy e o pandas, como no README do kit do curso. Essa aula não usa outra biblioteca.

```bash
python3 quatro_clientes.py
```

A saída é a da seção 2 da aula e sai igual toda vez.

```bash
python3 pca_clientes.py
python3 pca_clientes.py --sem-escala
python3 pca_clientes.py --componentes 2
python3 pca_clientes.py --componentes 2 --metades
```

| Opção | O que faz |
|---|---|
| `--colunas a,b,c` | Escolhe as colunas, separadas por vírgula e sem espaço. Sem ela, entram as oito colunas de comportamento. |
| `--sem-escala` | Só tira a média. Cada coluna fica na própria unidade, reais, dias ou porcentagem. |
| `--log a,b` | Passa essas colunas pelo log10 antes da escala. Só serve para coluna sempre positiva. |
| `--componentes k` | Reconstrói a tabela com k componentes e mostra o que volta de cada coluna e os cinco clientes que voltam pior. |
| `--metades` | Sorteia 20 vezes duas metades dos clientes, faz o PCA em cada uma e mede o ângulo entre as direções. Leva alguns segundos. |
| `--sem-maiores-gastos N` | Tira os N clientes de maior gasto antes de tudo. |
| `--sem-vazios` | Tira os clientes com valor vazio nas colunas escolhidas. Só precisa dela com `avaliacao_media`. |

As opções se combinam, por exemplo `python3 pca_clientes.py --sem-maiores-gastos 30 --componentes 2 --metades`. Os sorteios do `--metades` saem iguais toda vez.

Se o programa parar com `erro:`, a mensagem diz o que mudar: coluna que não existe, coluna de texto, valor vazio, log de zero ou um k fora do número de colunas.

Se aparecer `não achei ../dados/clientes/clientes.csv`, você está rodando de outra pasta. Entre na pasta dessa aula e rode de novo.
