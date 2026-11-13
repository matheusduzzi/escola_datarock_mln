# Kit do curso Aprendizado não supervisionado

Os arquivos que você usa nas aulas: o diagnóstico, os clientes da Loja Exemplo e as compras com cartão da mesma loja.

## Como o kit está organizado

| Pasta | O que tem |
|---|---|
| `00-apresentacao/` | O [diagnóstico](00-apresentacao/diagnostico.md), os programas das tarefas, a [folha de respostas](00-apresentacao/minhas-respostas.md), a [folha da ordem de estudo](00-apresentacao/minha-ordem.md) e a `primeira_olhada.py` da apresentação. |
| `dados/clientes/` | 3000 clientes com o que cada um comprou em 12 meses, sem coluna de grupo. Ver [dados/clientes/LEIA-ME.md](dados/clientes/LEIA-ME.md). |
| `dados/fraude/` | 8000 compras com cartão, as mesmas de ML supervisionado, com a contestação de fraude que o banco aceitou depois. Ver [dados/fraude/LEIA-ME.md](dados/fraude/LEIA-ME.md). |
| `NN-titulo/` | Uma pasta por aula, com o número e o título da aula. Elas entram no kit junto com as aulas. |

Os comandos das aulas rodam de dentro da pasta da aula. Quando uma aula usa uma base compartilhada, ela diz o caminho, por exemplo `../dados/clientes/clientes.csv`.

## Do que você precisa

- Python 3.9 ou mais novo, com o pandas e o NumPy. É o mesmo ambiente de ML supervisionado; se você ainda não preparou o seu, a aula 38. Dependências e ambientes, de Python para dados, mostra como.
- Um editor de código e um terminal.

As bibliotecas de projeção, de agrupamento e de anomalia entram na primeira aula que usa cada uma. Essa aula traz a versão, o Python mínimo e o comando de instalação.

Na pasta do kit, instale o que o diagnóstico usa:

```bash
python3 -m pip install -r requirements.txt
```

E veja se funcionou:

```bash
cd 00-apresentacao
python3 tarefa4.py
```

A primeira linha da saída é `clientes: 3000`; o resto é a tarefa 4 do diagnóstico. Se aparecer `No module named 'pandas'`, o `pip` instalou em outro Python. Rode `python3 -m pip install pandas` com o mesmo `python3` que você usa para rodar o programa. No Windows, troque `python3` por `py`.

Para gerar de novo as bases compartilhadas, rode `python3 gerar.py` dentro de cada pasta de `dados/`. As saídas são:

| Pasta | Saída de `python3 gerar.py` |
|---|---|
| `dados/clientes/` | `clientes=3000 pedidos=17541 avaliacao_vazia=945` |
| `dados/fraude/` | `transacoes=8000 fraudes=101` |

Os geradores só usam a biblioteca padrão e saem iguais toda vez.

## Se precisar de ajuda

Deixe um comentário na aula. Conte o objetivo, o comando que você rodou, a saída que apareceu e o que já tentou, sem dados pessoais.

## Licença

A licença de uso e redistribuição do kit ainda vai ser definida pela Escola Datarock. Até lá, use os arquivos para estudar o curso. A Loja Exemplo é inventada, e todos os dados foram gerados pelos programas do próprio kit.

## Aulas com arquivos neste kit

| Pasta | Aula |
|---|---|
| [00-apresentacao/](00-apresentacao/) | Apresentação |
| [01-objetivo-sem-rotulo/](01-objetivo-sem-rotulo/) | Objetivo sem rótulo |
| [02-autovalores-autovetores-e-svd/](02-autovalores-autovetores-e-svd/) | Autovalores, autovetores e SVD |
| [03-pca/](03-pca/) | PCA |
| [04-kernel-pca-t-sne-e-umap/](04-kernel-pca-t-sne-e-umap/) | Kernel PCA, t-SNE e UMAP |
| [06-k-means-e-k-medoids/](06-k-means-e-k-medoids/) | K-means e K-medoids |
| [07-densidade-e-hierarquia/](07-densidade-e-hierarquia/) | Densidade e hierarquia |
| [08-metricas-internas/](08-metricas-internas/) | Métricas internas |
| [09-segmentar-clientes/](09-segmentar-clientes/) | Segmentar clientes |
| [10-estabilidade-e-acao/](10-estabilidade-e-acao/) | Estabilidade e ação |
| [11-isolation-forest-lof-e-one-class-svm/](11-isolation-forest-lof-e-one-class-svm/) | Isolation Forest, LOF e One-Class SVM |
| [12-priorizar-suspeitas-de-fraude/](12-priorizar-suspeitas-de-fraude/) | Priorizar suspeitas de fraude |
| [13-avaliar-com-feedback/](13-avaliar-com-feedback/) | Avaliar com feedback |
| [14-defender-segmentacao-e-alertas/](14-defender-segmentacao-e-alertas/) | Defender segmentação e alertas |
