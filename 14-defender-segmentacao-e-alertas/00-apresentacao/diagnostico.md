# Diagnóstico: por onde começar em Aprendizado não supervisionado

A gerente de marketing da Loja Exemplo quer separar os clientes em grupos para mandar campanhas diferentes. A equipe de fraude quer outra coisa: uma fila das compras com cartão mais estranhas para revisar primeiro. Nenhum dos dois pedidos vem com resposta certa na base. Antes de rodar o primeiro agrupamento, eu refiz cinco contas que o curso usa desde a primeira aula. As cinco tarefas abaixo são essas contas e o código que eu usei.

Elas pedem o que vem dos cursos anteriores: escala e distância, NumPy sem loop e com teste, média e mediana com valores extremos e taxa-base de um evento raro. A tarefa 3 é de álgebra linear; se ela travar, tudo bem, porque a aula 02 ensina essa parte do zero. O diagnóstico não é prova e não tem nota. Ele mostra por onde você começa. Anote na [folha de respostas](minhas-respostas.md) a conta que fez, o comando que rodou e a saída que apareceu. Só depois compare com as respostas.

As tarefas 1, 3 e 5 são de lápis e papel ou calculadora. As tarefas 2 e 4 rodam de dentro da pasta `00-apresentacao/`:

```bash
cd 00-apresentacao
python3 tarefa4.py
```

No Windows, troque `python3` por `py`.

## Tarefa 1. Quem é o cliente mais parecido

Três clientes da Loja Exemplo, com o gasto e o número de pedidos dos últimos 12 meses:

| Cliente | Gasto (R$) | Pedidos |
|---|---|---|
| A | 1.200 | 2 |
| B | 1.150 | 12 |
| C | 300 | 3 |

Na base inteira, o desvio padrão do gasto é R$ 500 e o dos pedidos é 4.

1. Calcule a distância euclidiana de A até B e de A até C com os números como estão. Quem é o mais parecido com A?
2. Agora divida o gasto por 500 e os pedidos por 4 e calcule de novo. Quem é o mais parecido com A?
3. Qual das duas respostas está certa?
4. Volte aos números sem divisão, mas com o gasto em milhares de reais (1,2; 1,15; 0,3). Quem fica mais perto de A? O que isso diz sobre usar as colunas como estão?

## Tarefa 2. Escala e distâncias no NumPy

Em `tarefa2.py`, escreva duas funções, sem loop em Python:

| Função | O que retorna |
|---|---|
| `padronizar(X)` | um array do mesmo formato, com cada coluna na escala z: (x - média) / desvio padrão, com a média e o desvio de cada coluna; `X` pode chegar como lista de listas; para com `ValueError` se uma coluna tiver todos os valores iguais |
| `distancias(X)` | a matriz n x n das distâncias euclidianas entre as linhas de `X` |

Depois rode:

```bash
python3 tarefa2.py
python3 testar_tarefa2.py
```

Enquanto as funções não existem, os dois comandos param com `NotImplementedError`. Anote na folha:

1. As duas linhas que `tarefa2.py` mostrou.
2. No exemplo, o cliente 1 gasta R$ 40 a mais que o cliente 0, e o cliente 2 faz 4 pedidos a mais. Sem escala, quem fica mais perto do cliente 0? E depois da escala?
3. Por que `padronizar` precisa parar numa coluna com todos os valores iguais?
4. A sua função `distancias` monta um array de que formato no meio da conta, para 3.000 clientes e 10 colunas? Quantos números são?

## Tarefa 3. Uma matriz que estica o plano

A matriz abaixo leva cada ponto (x, y) do plano para outro ponto:

```text
A = | 2  1 |
    | 1  2 |
```

1. Calcule A vezes (1, 0), A vezes (1, 1) e A vezes (1, -1).
2. Quais desses vetores saem na mesma direção em que entraram? Esticados quantas vezes?
3. Desenhe o círculo de raio 1 e pense no que a matriz faz com ele. Em que direção ele fica mais comprido? Em qual fica mais curto?
4. Suponha que uma nuvem de pontos tenha variância 3 na direção de (1, 1) e 1 na direção de (1, -1). Que fração da variância total fica na direção mais comprida?

Se quiser, veja as respostas das perguntas 2 e 4 no NumPy com `np.linalg.eigh`.

## Tarefa 4. O cliente típico e os clientes do topo

Rode `python3 tarefa4.py`. Ele lê os 3.000 clientes de `../dados/clientes/clientes.csv`. Depois mostra o gasto médio e o mediano, quanto os 30 maiores somam e o desvio padrão com e sem eles. No fim vêm os cinco clientes de maior gasto. Anote a saída.

1. Para dizer quanto gasta um cliente típico, você usaria a média ou a mediana? Por que as duas ficaram tão longe uma da outra?
2. Olhe os cinco clientes do topo. O que eles têm em comum? O que você perguntaria à loja antes de decidir o que fazer com eles?
3. Um agrupamento que calcula o centro de cada grupo pela média recebe esses 30 clientes. O que eles fazem com o centro do grupo em que caírem?
4. Você padroniza o gasto com o desvio padrão que inclui os 30. O que acontece com a diferença entre os outros 2.970 clientes nessa coluna?

## Tarefa 5. A fila de revisão de fraude

No primeiro semestre de 2026, a Loja Exemplo teve 8.000 compras com cartão. Meses depois, o banco aceitou a contestação de fraude de 101 delas. Mas, na hora de revisar, ninguém sabia quais eram. A regra da equipe era revisar as 200 compras com o endereço de entrega mais longe do cadastro. Nessas 200, a equipe achou 30 fraudes.

1. Que fração das 200 compras revisadas era fraude? E das 8.000? Quantas vezes maior?
2. A equipe escreveu: "a regra pega 30 fraudes". Só com as 200 revisadas, dá para saber quantas fraudes ela deixou passar?
3. Para saber, a equipe sorteou 400 das 7.800 compras que a regra não marcou e revisou todas. Achou 2 fraudes. Estime quantas fraudes passaram e que fração das fraudes a regra pegou. Quanto você confia nessa estimativa, feita com 2 casos?
4. Um score novo foi testado só nas 200 compras que a regra antiga escolheu, porque eram as únicas com resposta. Entre elas, ele marcou mais fraudes. Por que essa comparação não mostra que o score novo é melhor?

## Depois das tarefas

Compare a folha com as respostas na aula e veja a tabela da apresentação do curso. Ela diz que aula estudar para cada tarefa em que você travou.

Se precisar de ajuda, deixe um comentário na aula. Conte o objetivo, o comando que você rodou, a saída que apareceu e o que já tentou, sem dados pessoais.
