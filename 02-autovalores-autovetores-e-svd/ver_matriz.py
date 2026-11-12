"""Mostra o que uma matriz 2 x 2 faz com o plano. Use para ver se a sua conta à mão bateu.

    python3 ver_matriz.py 2 1 1 2
    python3 ver_matriz.py 2 1 1 2 --desenho minha_matriz.png

Os quatro números são a matriz linha por linha: a b c d quer dizer

    | a  b |
    | c  d |

Precisa do NumPy. O --desenho precisa também do matplotlib.
"""
import math
import sys

import numpy as np

TOL = 1e-9


def num(x, casas=2):
    """Número com vírgula, como no texto da aula. Zero sem sinal."""
    if abs(x) < 0.5 * 10 ** (-casas):
        x = 0.0
    s = f'{x:.{casas}f}'.replace('.', ',')
    return s


def vet(v, casas=2):
    return f'({num(v[0], casas)}; {num(v[1], casas)})'


def sentido(v):
    """Escolhe o sinal do vetor: o primeiro número diferente de zero fica positivo."""
    v = np.asarray(v, dtype=float)
    i = 0 if abs(v[0]) > TOL else 1
    return v if v[i] > 0 else -v


def angulo(r):
    """Ângulo, em graus, de uma matriz de rotação 2 x 2."""
    return math.degrees(math.atan2(r[1, 0], r[0, 0]))


def ler_matriz(args):
    if len(args) != 4:
        raise ValueError(f'preciso de 4 números, a b c d, e recebi {len(args)}')
    try:
        a, b, c, d = (float(x.replace(',', '.')) for x in args)
    except ValueError:
        raise ValueError(f'não consegui ler {args} como números') from None
    return np.array([[a, b], [c, d]])


def direcoes(m):
    """Autovalores e autovetores reais, do maior autovalor para o menor.

    Retorna (lista de (autovalor, autovetor), mensagem). A lista fica vazia quando nenhuma direção real
    fica no lugar."""
    if np.allclose(m, m[0, 0] * np.eye(2), atol=TOL):
        return [(m[0, 0], np.array([1.0, 0.0])), (m[0, 0], np.array([0.0, 1.0]))], \
            'toda direção só estica: a matriz multiplica qualquer vetor pelo mesmo número'
    if np.allclose(m, m.T, atol=TOL):
        valores, vetores = np.linalg.eigh(m)  # simétrica: autovalores reais, em ordem crescente
    else:
        valores, vetores = np.linalg.eig(m)
    if np.any(np.abs(np.imag(valores)) > TOL):
        return [], 'nenhuma direção real fica no lugar: a matriz gira todo vetor'
    valores, vetores = np.real(valores), np.real(vetores)
    ordem = np.argsort(-valores, kind='stable')
    pares = [(valores[i], sentido(vetores[:, i])) for i in ordem]
    if abs(np.linalg.det(np.column_stack([p[1] for p in pares]))) < 1e-6:
        return pares[:1], 'só uma direção fica no lugar: as duas colunas do NumPy apontam para o mesmo lado'
    return pares, ''


def svd_girar_esticar_girar(m):
    """M = U @ diag(s) @ V.T com V uma rotação de -90 a 90 graus. U é rotação ou espelho seguido de rotação."""
    u, s, vt = np.linalg.svd(m)
    v = vt.T
    if np.linalg.det(v) < 0:  # troca o sentido do segundo par: o produto não muda
        v[:, 1] *= -1
        u[:, 1] *= -1
    if not -90 < angulo(v) <= 90:  # gira os dois por meia volta: o produto não muda
        v, u = -v, -u
    return u, s, v


def efeito(valor):
    if abs(valor) < TOL:
        return '   (achata: essa direção inteira vai para o ponto zero)'
    if valor < 0:
        return '   (inverte o sentido)'
    return ''


def relatorio(m):
    a, b, c, d = m.ravel()
    linhas = ['matriz', f'  | {num(a)}  {num(b)} |', f'  | {num(c)}  {num(d)} |',
              'para onde vão os eixos (são as colunas da matriz)',
              f'  (1; 0) -> {vet(m[:, 0])}', f'  (0; 1) -> {vet(m[:, 1])}']
    det = np.linalg.det(m)
    lado = ', com o lado trocado (espelho)' if det < -TOL else ''
    if abs(det) < TOL:
        linhas.append(f'determinante {num(det)}: o quadrado de lado 1 vira um segmento, de área zero')
    else:
        linhas.append(f'determinante {num(det)}: o quadrado de lado 1 vira um paralelogramo de área {num(abs(det))}{lado}')
    linhas.append('direções que só esticam (autovetores) e quanto esticam (autovalores)')
    pares, aviso = direcoes(m)
    for valor, vetor in pares:
        linhas.append(f'  {vet(vetor)} x {num(valor)}{efeito(valor)}')
    if aviso:
        linhas.append(f'  {aviso}')
    if len(pares) == 2 and not aviso:
        cos = float(pares[0][1] @ pares[1][1]) / (np.linalg.norm(pares[0][1]) * np.linalg.norm(pares[1][1]))
        linhas.append(f'  ângulo entre as duas direções: {num(math.degrees(math.acos(max(-1, min(1, abs(cos))))), 1)} graus')
    u, s, v = svd_girar_esticar_girar(m)
    linhas.append('o círculo de raio 1 vira uma elipse (SVD)')
    linhas.append(f'  raio maior {num(s[0])} na direção {vet(sentido(u[:, 0]))}')
    linhas.append(f'  raio menor {num(s[1])} na direção {vet(sentido(u[:, 1]))}')
    linhas.append('SVD em três passos: M = U x S x Vt')
    linhas.append(f'  1. Vt gira o plano {num(-angulo(v), 1)} graus')
    linhas.append(f'  2. S estica {num(s[0])} vezes no eixo x e {num(s[1])} vezes no eixo y')
    if np.linalg.det(u) < 0:
        linhas.append(f'  3. U espelha no eixo x e gira {num(angulo(u @ np.diag([1, -1])), 1)} graus')
    else:
        linhas.append(f'  3. U gira o plano {num(angulo(u), 1)} graus')
    refeita = u @ np.diag(s) @ v.T
    linhas.append(f'  U x S x Vt refaz a matriz: {"sim" if np.allclose(refeita, m) else "NÃO"}')
    linhas.append('oito pontos do círculo e para onde vão')
    for graus in range(0, 360, 45):
        p = np.array([math.cos(math.radians(graus)), math.sin(math.radians(graus))])
        linhas.append(f'  {graus:3d} graus {vet(p)} -> {vet(m @ p)}')
    return '\n'.join(linhas)


def desenhar(m, arquivo):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        raise SystemExit('o --desenho precisa do matplotlib: python3 -m pip install matplotlib')
    t = np.linspace(0, 2 * math.pi, 361)
    circulo = np.vstack([np.cos(t), np.sin(t)])
    elipse = m @ circulo
    quadrado = np.array([[0, 1, 1, 0, 0], [0, 0, 1, 1, 0]], dtype=float)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(*circulo, color='#9aa5a0', label='círculo de raio 1')
    ax.plot(*elipse, color='#52645C', linewidth=2, label='depois da matriz')
    ax.fill(*quadrado, color='#9aa5a0', alpha=0.25, label='quadrado de lado 1')
    ax.fill(*(m @ quadrado), color='#52645C', alpha=0.25, label='paralelogramo')
    for valor, vetor in direcoes(m)[0]:
        w = vetor / np.linalg.norm(vetor)
        comprido = max(3.0, abs(valor) * 1.3)
        ax.plot([-comprido * w[0], comprido * w[0]], [-comprido * w[1], comprido * w[1]], '--', color='#c0392b', linewidth=1)
        ax.annotate(f'x {num(valor)}', valor * w, color='#c0392b')
    lim = max(1.5, float(np.abs(elipse).max()), float(np.abs(m @ quadrado).max())) * 1.15
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')
    ax.axhline(0, color='#cccccc', linewidth=0.8)
    ax.axvline(0, color='#cccccc', linewidth=0.8)
    ax.legend(loc='upper left', fontsize=8)
    ax.set_title('tracejado: direções que só esticam')
    fig.savefig(arquivo, dpi=120, bbox_inches='tight')
    return arquivo


def main(argv):
    args = list(argv)
    arquivo = None
    if '--desenho' in args:
        i = args.index('--desenho')
        if i + 1 >= len(args):
            print('erro: depois de --desenho vem o nome do arquivo, por exemplo minha_matriz.png', file=sys.stderr)
            return 2
        arquivo = args[i + 1]
        del args[i:i + 2]
    try:
        m = ler_matriz(args)
    except ValueError as erro:
        print(f'erro: {erro}. Exemplo: python3 ver_matriz.py 2 1 1 2', file=sys.stderr)
        return 2
    print(relatorio(m))
    if arquivo:
        print(f'desenho salvo em {desenhar(m, arquivo)}')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
