# Minhas respostas: Métricas internas

Escreva aqui as respostas da prática guiada e do exercício. Nas contas à mão, mostre cada distância e cada média, não só o resultado.

## Prática guiada

### Pedido 1. O s do 23 com o ticket em reais

Resposta:

### Pedido 2. A proposta da agência medida na escala z

Grupo com a pior silhouette:

Clientes abaixo de 0:

Por quê:

### Pedido 3. O grupo do Ward com a menor silhouette

Grupo:

Isso quer dizer que ele deve sair da campanha?

## Exercício: a nota do Rafael vale?

### 1. Os clientes

Cole aqui a tabela dos clientes e a tabela de distâncias de `python3 a_mao.py --ids 33,138,91,28,39,34,131,2016`.

### 2. A silhouette à mão

**Cliente 39**

| | Distâncias que entram | Média |
|---|---|---|
| a (o grupo dele) | | |
| b (o outro grupo) | | |

Grupo vizinho:

s = ( b - a ) / maior dos dois =

O que esse s quer dizer:

**Cliente 2016**

| | Distâncias que entram | Média |
|---|---|---|
| a (o grupo dele) | | |
| b (o outro grupo) | | |

Grupo vizinho:

s =

O que esse s quer dizer:

### 3. O Davies-Bouldin à mão

Centro de "compra pouco" (pedidos; ticket):

Centro de "compra muito" (pedidos; ticket):

Conta inteira de uma distância até o centro (diferenças, quadrados, soma, raiz):

| Grupo | Cliente | Distância até o centro |
|---|---|---|
| compra pouco | 33 | |
| compra pouco | 138 | |
| compra pouco | 91 | |
| compra pouco | 28 | |
| compra muito | 39 | |
| compra muito | 34 | |
| compra muito | 131 | |
| compra muito | 2016 | |

Espalhamento de "compra pouco":

Espalhamento de "compra muito":

Distância entre os centros:

Davies-Bouldin:

### 4. Veja se bateu

O que bateu, o que deu diferente e por quê:

Silhouette média:

Calinski-Harabasz:

### 5. As três divisões

| Divisão | silhouette | Davies-Bouldin | Calinski-Harabasz |
|---|---|---|---|
| Júlia: 33, 138, 91, 28 / 39, 34, 131, 2016 | | | |
| Rafael: 33, 138, 91, 28, 39, 34, 131 / 2016 | | | |
| Três grupos: 33, 138, 91, 28 / 39, 34, 131 / 2016 | | | |

Qual divisão cada nota prefere:

A divisão do Rafael separa quem compra muito de quem compra pouco? Por que a nota dela é maior?

### 6. A régua

| Divisão | Ticket | silhouette | Davies-Bouldin | Calinski-Harabasz |
|---|---|---|---|---|
| Júlia | centenas | | | |
| Júlia | reais | | | |
| Rafael | centenas | | | |
| Rafael | reais | | | |

O que aconteceu e por quê:

### 7. A base toda

| Divisão | medida em | silhouette | Davies-Bouldin | Calinski-Harabasz |
|---|---|---|---|---|
| Rafael: K-means K = 4 no PCA | | | | |
| Ward com 4 grupos | | | | |
| K-means K = 4 na escala z | | | | |

Onde o 0,544 e o 0,353 foram medidos:

Medidas no mesmo lugar, qual divisão tem as melhores notas? A diferença é grande?

Pior grupo de cada divisão, com a silhouette e os clientes abaixo de 0:

### 8. A resposta para a Júlia

Resposta:
