# Minhas respostas: K-means e K-medoids

Escreva aqui as respostas da aula «K-means e K-medoids». Cole as linhas da saída que você usou em cada resposta.

## Prática guiada

**Pedido 1. O ticket em reais.**

**Pedido 2. A base toda sem os clientes do topo.**

**Pedido 3. Vinte rodadas com um começo só.**

## Exercício: dois grupos para a Júlia, feitos à mão

### 1. Os clientes

Cole a tabela de `python3 a_mao.py --ids 33,138,91,28,39,34,131,2016`.

### 2. O K-means à mão

Duas contas inteiras (diferença de pedidos, diferença de ticket, quadrados, soma, raiz). Uma delas do cliente 2016.

- Conta 1:
- Conta 2 (cliente 2016):

Rodada 1. Centros: A = cliente 33, B = cliente 39.

| Cliente | Distância a A | Distância a B | Grupo |
|---|---|---|---|
| 33 | | | |
| 138 | | | |
| 91 | | | |
| 28 | | | |
| 39 | | | |
| 34 | | | |
| 131 | | | |
| 2016 | | | |

Centros novos. A: pedidos ____, ticket ____. B: pedidos ____, ticket ____.

Rodada 2. Centros: os de cima.

| Cliente | Distância a A | Distância a B | Grupo |
|---|---|---|---|
| 33 | | | |
| 138 | | | |
| 91 | | | |
| 28 | | | |
| 39 | | | |
| 34 | | | |
| 131 | | | |
| 2016 | | | |

Se algum cliente trocou de grupo, copie a tabela e faça mais uma rodada.

O K-means parou na rodada ____. Grupo A: ____. Grupo B: ____.

### 3. O K-medoids à mão

Rodada 1. Medoides: A = cliente 33, B = cliente 39.

| Cliente | Distância a A | Distância a B | Grupo | Soma das distâncias até os outros do grupo |
|---|---|---|---|---|
| 33 | | | | |
| 138 | | | | |
| 91 | | | | |
| 28 | | | | |
| 39 | | | | |
| 34 | | | | |
| 131 | | | | |
| 2016 | | | | |

Medoides novos. A: ____. B: ____.

Se algum medoide mudou, copie a tabela e faça mais uma rodada.

Medoides do fim: ____. Grupos do fim: ____.

### 4. Veja se bateu

O que deu diferente do programa e por quê:

### 5. O cliente 2016

### 6. O K

### 7. A resposta para a Júlia
