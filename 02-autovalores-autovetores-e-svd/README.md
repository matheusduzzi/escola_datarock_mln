# 02. Autovalores, autovetores e SVD

Os arquivos da aula «Autovalores, autovetores e SVD» do curso Aprendizado não supervisionado. Rode os comandos de dentro dessa pasta.

| Arquivo | Para que serve |
|---|---|
| `transformacao.py` | As contas da aula, na ordem: a matriz que estica o plano, a matriz que gira e o SVD dela, e a covariância de pedidos e gasto dos clientes. |
| `ver_matriz.py` | Mostra o que qualquer matriz 2 × 2 faz com o plano: eixos, determinante, autovalores, autovetores, SVD e os oito pontos do círculo. Use para ver se a sua conta à mão bateu. |
| `duas_colunas.py` | Monta a matriz de covariância de duas colunas dos clientes, cada uma em escala z. Ele não calcula as direções: essa parte é sua. |
| `minhas-respostas.md` | Onde você escreve as contas e as respostas do exercício. |

Os clientes vêm de `../dados/clientes/clientes.csv`, a base do curso.

## Como rodar

Você precisa do Python com o NumPy e o pandas, como no README do kit do curso.

```bash
python3 transformacao.py
```

A saída é a mesma das seções 3, 5, 6 e 8 da aula, e sai igual toda vez.

```bash
python3 ver_matriz.py 2 1 1 2
```

Os quatro números são a matriz linha por linha. Pode usar decimal com ponto ou com vírgula: `0.87` ou `0,87`. Com menos ou mais de quatro números, ele para e mostra um exemplo.

```bash
python3 duas_colunas.py pedidos_12m gasto_12m
python3 duas_colunas.py pedidos_12m gasto_12m --sem-maiores-gastos 30
```

Os nomes das colunas são os do `LEIA-ME.md` dos clientes. Um nome errado faz ele parar e listar as colunas que servem.

## O desenho (opcional)

```bash
python3 ver_matriz.py 2 1 1 2 --desenho a.png
```

Salva o quadrado, o círculo, o paralelogramo, a elipse e as direções que só esticam num PNG. Precisa do matplotlib: `python3 -m pip install matplotlib`. Sem ele, o resto do programa funciona e só o `--desenho` avisa o que instalar. Faça o seu desenho à mão antes de gerar o PNG.

Se aparecer `FileNotFoundError` ou `não achei ../dados/clientes/clientes.csv`, você está rodando de outra pasta. Entre na pasta dessa aula e rode de novo.
