import numpy as np
import matplotlib.pyplot as plt
import io, base64, json

plt.rcParams.update({
    'figure.facecolor': '#111827', 'axes.facecolor': '#0a0e1a',
    'axes.edgecolor':   '#1e2d45', 'axes.labelcolor': '#94a3b8',
    'xtick.color':      '#64748b', 'ytick.color':     '#64748b',
    'text.color':       '#e2e8f0', 'grid.color':      '#1e2d45',
    'grid.linewidth':   0.6,       'axes.grid':       True,
    'legend.facecolor': '#111827', 'legend.edgecolor':'#1e2d45',
    'legend.fontsize':  8,
})

TEAL = '#00d4aa'; ORANGE = '#ff6b35'; MUTED = '#64748b'; TEXT = '#e2e8f0'

# ── Funções de ativação ────────────────────────────────────────────────────────
def sigmoid(z):    return 1 / (1 + np.exp(-z))
def d_sig(a):      return a * (1 - a)          # derivada dada a saída

def relu(z):       return np.maximum(0, z)
def d_relu(z):     return (z > 0).astype(float) # derivada dada a ENTRADA z

def _b64(fig):
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=90)
    s = 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()
    plt.close(fig)
    return s

# ── Diagrama da rede ───────────────────────────────────────────────────────────
NP = {'X':(0.5,0.62),'HA1':(1.9,0.85),'HA2':(1.9,0.38),
      'HB1':(3.3,0.85),'HB2':(3.3,0.38),'Y':(4.7,0.62)}
NC = {'X':'#a07820','HA1':'#2060a0','HA2':'#2060a0',
      'HB1':'#206840','HB2':'#206840','Y':'#802020'}
R  = 0.13

def _base_ax():
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.set_xlim(0, 5.4); ax.set_ylim(-0.05, 1.18); ax.axis('off')
    for lx, lb in [(0.5,'Entrada X'),(1.9,'Oculta A'),(3.3,'Oculta B'),(4.7,'Saída ŷ')]:
        ax.text(lx, 1.10, lb, ha='center', fontsize=8.5, color=MUTED, style='italic')
    return fig, ax

def _arrow(ax, src, dst, color, lbl, sgn):
    xs,ys=NP[src]; xe,ye=NP[dst]; dx,dy=xe-xs,ye-ys; d=np.sqrt(dx**2+dy**2)
    ax.annotate("",xy=(xe-(dx/d)*R,ye-(dy/d)*R),xytext=(xs+(dx/d)*R,ys+(dy/d)*R),
                arrowprops=dict(arrowstyle="->",color=color,lw=1.5,alpha=0.85))
    ax.text((xs+xe)/2,(ys+ye)/2+sgn*0.055,lbl,fontsize=6.5,ha='center',color=color,
            fontweight='bold',bbox=dict(boxstyle='round,pad=0.12',fc='#111827',ec='none',alpha=0.92))

def _nodes(ax, vals, delta=False):
    for n,(px,py) in NP.items():
        ax.add_artist(plt.Circle((px,py),R,color=NC[n],ec='#556',lw=1.5,zorder=4))
        top = n if (not delta or n=='X') else 'δ'
        ax.text(px,py+0.030,top,ha='center',va='center',fontsize=7.5,
                fontweight='bold',zorder=5,color='#cbd5e1')
        vc = ORANGE if (delta and n!='X') else TEXT
        ax.text(px,py-0.055,f"{vals.get(n,0):.4f}",ha='center',va='center',
                fontsize=7,color=vc,zorder=5)

# ── Gráficos de funções de ativação ───────────────────────────────────────────
def plot_sigmoid():
    z = np.linspace(-6, 6, 300); a = sigmoid(z)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].plot(z, a, color=TEAL, lw=2.5)
    axes[0].axhline(0.5, color=MUTED, ls='--', lw=1, label='y = 0.5')
    axes[0].axvline(0,   color=MUTED, ls='--', lw=1, label='z = 0')
    axes[0].set_title('Sigmoide  σ(z)', fontsize=12, color=TEXT)
    axes[0].set_xlabel('z'); axes[0].set_ylabel('σ(z)'); axes[0].set_ylim(-0.05,1.05); axes[0].legend()
    axes[1].plot(z, d_sig(a), color=ORANGE, lw=2.5)
    axes[1].fill_between(z, d_sig(a), alpha=0.15, color=ORANGE)
    axes[1].set_title("Derivada  σ'(z) = σ(z)·(1−σ(z))", fontsize=11, color=TEXT)
    axes[1].set_xlabel('z'); axes[1].set_ylabel("σ'(z)")
    axes[1].axhline(0.25, color=MUTED, ls='--', lw=1, label='máx = 0.25')
    axes[1].legend()
    plt.tight_layout(); return _b64(fig)

def plot_relu():
    z = np.linspace(-4, 4, 300)
    r = relu(z); dr = d_relu(z)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))

    # ReLU
    axes[0].plot(z, r, color=TEAL, lw=2.5)
    axes[0].fill_between(z, r, where=(z>0), alpha=0.12, color=TEAL)
    axes[0].axvline(0, color=MUTED, ls='--', lw=1)
    axes[0].set_title('ReLU  f(z) = max(0, z)', fontsize=12, color=TEXT)
    axes[0].set_xlabel('z'); axes[0].set_ylabel('ReLU(z)')

    # Anotações
    axes[0].annotate('zona morta\n(saída = 0)', xy=(-2, 0),
                     xytext=(-3.5, 0.8), color=ORANGE, fontsize=8,
                     arrowprops=dict(arrowstyle='->', color=ORANGE, lw=1.2))
    axes[0].annotate('zona ativa\n(saída = z)', xy=(2, 2),
                     xytext=(0.5, 3), color=TEAL, fontsize=8,
                     arrowprops=dict(arrowstyle='->', color=TEAL, lw=1.2))

    # Derivada (degrau)
    axes[1].step(z, dr, color=ORANGE, lw=2.5, where='post')
    axes[1].fill_between(z, dr, alpha=0.15, color=ORANGE)
    axes[1].set_title("Derivada  f'(z) = 0 se z≤0,  1 se z>0", fontsize=11, color=TEXT)
    axes[1].set_xlabel('z'); axes[1].set_ylabel("f'(z)")
    axes[1].set_ylim(-0.1, 1.4)
    axes[1].axhline(1, color=MUTED, ls='--', lw=1, label="f'(z) = 1  (z > 0)")
    axes[1].axhline(0, color=MUTED, ls=':',  lw=1, label="f'(z) = 0  (z ≤ 0)")
    axes[1].legend()

    plt.tight_layout(); return _b64(fig)

def plot_comparison():
    """Gráfico comparativo Sigmoide × ReLU."""
    z = np.linspace(-4, 4, 300)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))

    axes[0].plot(z, sigmoid(z), color=TEAL,   lw=2.2, label='Sigmoide')
    axes[0].plot(z, relu(z),    color=ORANGE, lw=2.2, label='ReLU')
    axes[0].set_title('Saída das funções', fontsize=11, color=TEXT)
    axes[0].set_xlabel('z'); axes[0].legend(); axes[0].set_ylim(-0.2, 4.2)

    axes[1].plot(z, d_sig(sigmoid(z)), color=TEAL,   lw=2.2, label="σ'  (máx 0.25)")
    axes[1].plot(z, d_relu(z),         color=ORANGE, lw=2.2, label="ReLU'  (0 ou 1)")
    axes[1].set_title('Gradientes', fontsize=11, color=TEXT)
    axes[1].set_xlabel('z'); axes[1].legend()
    axes[1].annotate('gradiente\ndesaparece', xy=(-3.5, d_sig(sigmoid(-3.5))),
                     xytext=(-3.8, 0.12), color=TEAL, fontsize=8,
                     arrowprops=dict(arrowstyle='->', color=TEAL, lw=1))

    plt.suptitle('Sigmoide vs ReLU', color=TEXT, fontsize=12)
    plt.tight_layout(); return _b64(fig)

def plot_forward(X, hA, hB, yp, W1, W2, W3, title):
    fig,ax=_base_ax()
    conns=[('X','HA1',f"w={W1[0]:.3f}",+1),('X','HA2',f"w={W1[1]:.3f}",-1),
           ('HA1','HB1',f"w={W2[0,0]:.3f}",+1),('HA1','HB2',f"w={W2[0,1]:.3f}",-1),
           ('HA2','HB1',f"w={W2[1,0]:.3f}",+1),('HA2','HB2',f"w={W2[1,1]:.3f}",-1),
           ('HB1','Y',f"w={W3[0]:.3f}",+1),('HB2','Y',f"w={W3[1]:.3f}",-1)]
    for s,d,lb,sg in conns: _arrow(ax,s,d,TEAL,lb,sg)
    _nodes(ax,{'X':X,'HA1':hA[0],'HA2':hA[1],'HB1':hB[0],'HB2':hB[1],'Y':yp})
    ax.set_title(title,fontsize=10.5,color=TEXT,pad=8)
    plt.tight_layout(); return _b64(fig)

def plot_backprop(X, dW1, dW2, dW3, dhA, dhB, dY, title):
    fig,ax=_base_ax()
    conns=[('Y','HB1',f"∇={dW3[0]:.4f}",+1),('Y','HB2',f"∇={dW3[1]:.4f}",-1),
           ('HB1','HA1',f"∇={dW2[0,0]:.4f}",+1),('HB2','HA1',f"∇={dW2[0,1]:.4f}",-1),
           ('HB1','HA2',f"∇={dW2[1,0]:.4f}",+1),('HB2','HA2',f"∇={dW2[1,1]:.4f}",-1),
           ('HA1','X',f"∇={dW1[0]:.4f}",+1),('HA2','X',f"∇={dW1[1]:.4f}",-1)]
    for s,d,lb,sg in conns: _arrow(ax,s,d,ORANGE,lb,sg)
    _nodes(ax,{'X':X,'HA1':dhA[0],'HA2':dhA[1],'HB1':dhB[0],'HB2':dhB[1],'Y':dY},delta=True)
    ax.set_title(title,fontsize=10.5,color=TEXT,pad=8)
    plt.tight_layout(); return _b64(fig)

def plot_learning_curve(hist_err, hist_pred, Y, lr, X, epochs, act_name):
    ep = list(range(1, len(hist_err)+1))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(ep, hist_err, color=ORANGE, lw=2)
    marks = sorted(set([0, min(9,len(hist_err)-1), len(hist_err)//2, len(hist_err)-1]))
    axes[0].scatter([ep[m] for m in marks],[hist_err[m] for m in marks],
                    color='#c0392b', zorder=5, s=55)
    axes[0].set_title('Curva de Aprendizado — Erro MSE', fontsize=11, color=TEXT)
    axes[0].set_xlabel('Épocas'); axes[0].set_ylabel('Erro (MSE)')
    axes[1].plot(ep, hist_pred, color=TEAL, lw=2, label='Predição ŷ')
    axes[1].axhline(Y, color='#27ae60', ls='--', lw=2, label=f'Alvo y = {Y}')
    axes[1].set_title('Predição convergindo para o alvo', fontsize=11, color=TEXT)
    axes[1].set_xlabel('Épocas'); axes[1].set_ylabel('ŷ'); axes[1].legend()
    plt.suptitle(f'Treinamento: X={X}, alvo={Y}, lr={lr}, {epochs} épocas  [{act_name}]',
                 color=TEXT, fontsize=11)
    plt.tight_layout(); return _b64(fig)


# ── Função principal ───────────────────────────────────────────────────────────
def nn_run_all(X_s,Y_s,lr_s,w1_0,w1_1,w2_00,w2_01,w2_10,w2_11,w3_0,w3_1,epochs_s,act_s="sigmoid"):
    try:
        X=float(X_s); Y=float(Y_s); lr=float(lr_s); epochs=int(epochs_s)
        W1=np.array([float(w1_0),float(w1_1)])
        W2=np.array([[float(w2_00),float(w2_01)],[float(w2_10),float(w2_11)]])
        W3=np.array([float(w3_0),float(w3_1)])

        use_relu = (act_s == "relu")
        act_name = "ReLU" if use_relu else "Sigmoide"

        # helpers que respeitam a função escolhida nas camadas ocultas
        # (saída sempre usa sigmoide para manter ŷ ∈ (0,1))
        def act(z):   return relu(z)    if use_relu else sigmoid(z)
        def dact(z):  return d_relu(z)  if use_relu else d_sig(sigmoid(z))
        # nota: para sigmoide passamos a SAÍDA; para relu passamos z
        # unificamos: dact recebe SEMPRE z (pre-ativação)
        def dact_from_z(z):
            return d_relu(z) if use_relu else d_sig(sigmoid(z))

        steps = []

        # ── Passo 0: Configuração ────────────────────────────────────────
        steps.append({"title":"Configuração da Rede","sections":[
            {"type":"text","content":
             "Vamos construir uma MLP com 4 camadas: Entrada (1 neurônio), Oculta A (2), Oculta B (2) e Saída (1). "
             f"As camadas ocultas usarão {act_name}; a saída usa sempre Sigmoide para manter ŷ ∈ (0, 1)."},
            {"type":"math","content":
             r"X \xrightarrow{W_1} h_A \xrightarrow{W_2} h_B \xrightarrow{W_3} \hat{y}"},
            {"type":"subtitle","content":"Por que precisamos de pesos iniciais?"},
            {"type":"text","content":
             "Os pesos são os parâmetros que a rede vai aprender. Antes do treinamento, precisamos de valores de partida. "
             "Não podemos começar com todos em zero: se W=0, todos os neurônios calculam a mesma coisa e o gradiente também é zero — "
             "a rede nunca sairia do lugar (problema da simetria). "
             "Valores diferentes 'quebram' essa simetria e permitem que cada neurônio especialize sua função."},
            {"type":"subtitle","content":"O que cada matriz de pesos conecta"},
            {"type":"table","headers":["Matriz","Dimensão","Conecta","Interpretação"],"rows":[
                ["W1","1 × 2","Entrada → Oculta A",
                 f"W1[0]={w1_0} escala X para hA1;  W1[1]={w1_1} escala X para hA2"],
                ["W2","2 × 2","Oculta A → Oculta B",
                 "Cada coluna de W2 combina hA1 e hA2 para produzir um neurônio de hB"],
                ["W3","2 × 1","Oculta B → Saída",
                 f"W3[0]={w3_0} pondera hB1;  W3[1]={w3_1} pondera hB2"],
            ]},
            {"type":"subtitle","content":"Hiperparâmetros definidos"},
            {"type":"table","headers":["Parâmetro","Valor","Papel"],"rows":[
                ["Entrada X",   str(X),       "Valor que alimenta a rede"],
                ["Alvo y",      str(Y),       "Saída que queremos que a rede aprenda a produzir"],
                ["lr",          str(lr),      "Tamanho do passo de ajuste. Alto → instável; baixo → lento"],
                ["Épocas",      str(epochs),  "Quantas vezes repetimos o ciclo forward→backprop→atualização"],
                ["Ativação",    act_name,     "Função usada nas camadas ocultas"],
            ]},
            {"type":"highlight",
             "content":f"Objetivo: dado X = {X}, a rede deve ajustar W1, W2, W3 até produzir ŷ ≈ {Y}.","variant":"teal"},
        ]})

        # ── Passo 1: Função de Ativação ───────────────────────────────────
        if use_relu:
            steps.append({"title":"Função de Ativação: ReLU","sections":[
                {"type":"text","content":
                 "A ReLU (Rectified Linear Unit) é a função de ativação mais usada em redes profundas modernas. "
                 "Ela é matematicamente simples: retorna zero para entradas negativas e a própria entrada para positivas."},
                {"type":"math","content": r"f(z) = \max(0,\, z) = \begin{cases} 0 & z \leq 0 \\ z & z > 0 \end{cases}"},
                {"type":"text","content":
                 "Sua derivada é ainda mais simples — um degrau: vale 0 para z negativo e 1 para z positivo. "
                 "Isso resolve o problema do vanishing gradient que a sigmoide tem: "
                 "o gradiente na zona ativa nunca diminui ao ser propagado para trás."},
                {"type":"math","content": r"f'(z) = \begin{cases} 0 & z \leq 0 \\ 1 & z > 0 \end{cases}"},
                {"type":"img","content": plot_relu()},
                {"type":"subtitle","content":"Zona morta (dying ReLU)"},
                {"type":"text","content":
                 "O maior risco da ReLU é a 'zona morta': se z ≤ 0, o gradiente é exatamente 0. "
                 "Um neurônio pode ficar preso nessa zona e parar de aprender permanentemente, "
                 "especialmente com learning rates muito altos que 'empurram' z para negativo. "
                 "Nesta demonstração, como temos apenas 2 épocas detalhadas e pesos pequenos, "
                 "esse problema raramente aparece — mas é importante conhecê-lo."},
                {"type":"table","headers":["z (entrada)","ReLU(z)","f'(z) (gradiente)"],"rows":[
                    ["-3", "0.0000", "0  ← gradiente zero (zona morta)"],
                    ["-1", "0.0000", "0  ← zona morta"],
                    ["0",  "0.0000", "0  ← limiar"],
                    ["0.5",f"{relu(np.array([0.5]))[0]:.4f}", "1  ← gradiente pleno"],
                    ["2",  "2.0000", "1  ← gradiente pleno"],
                    ["5",  "5.0000", "1  ← sem saturação (ao contrário da sigmoide)"],
                ]},
                {"type":"img","content": plot_comparison()},
                {"type":"highlight",
                 "content":"Vantagem chave da ReLU sobre a Sigmoide: gradiente constante (=1) na zona ativa. "
                           "A sigmoide satura para z grande, causando gradientes próximos de zero e aprendizado lento.",
                 "variant":"teal"},
            ]})
        else:
            steps.append({"title":"Função de Ativação: Sigmoide","sections":[
                {"type":"text","content":
                 "Sem uma função de ativação, a rede inteira seria apenas uma transformação linear — não importaria quantas "
                 "camadas houvesse, o resultado seria equivalente a uma única multiplicação de matrizes. "
                 "A sigmoide introduz não-linearidade: ela 'dobra' o espaço de forma que padrões complexos possam ser aprendidos."},
                {"type":"math","content": r"\sigma(z) = \frac{1}{1 + e^{-z}} \quad \in (0,\,1)"},
                {"type":"text","content":
                 "Sua derivada é elegante porque pode ser expressa em termos da própria saída — "
                 "o que torna o backpropagation eficiente: se já calculamos σ(z) = a no forward pass, "
                 "basta fazer a·(1−a). Não precisamos rearmazenar z."},
                {"type":"math","content": r"\sigma'(z) = \sigma(z)\cdot\bigl(1-\sigma(z)\bigr) \quad \leq\, 0.25"},
                {"type":"img","content": plot_sigmoid()},
                {"type":"table","headers":["z","σ(z)","σ'(z)","Observação"],"rows":[
                    ["-4", f"{sigmoid(-4):.4f}", f"{d_sig(sigmoid(-4)):.4f}", "quase zero — gradiente mínimo"],
                    ["-1", f"{sigmoid(-1):.4f}", f"{d_sig(sigmoid(-1)):.4f}", ""],
                    ["0",  f"{sigmoid(0):.4f}",  f"{d_sig(sigmoid(0)):.4f}",  "derivada máxima = 0.25"],
                    ["1",  f"{sigmoid(1):.4f}",  f"{d_sig(sigmoid(1)):.4f}",  ""],
                    ["4",  f"{sigmoid(4):.4f}",  f"{d_sig(sigmoid(4)):.4f}",  "quase zero — gradiente mínimo"],
                ]},
                {"type":"img","content": plot_comparison()},
                {"type":"highlight",
                 "content":"Atenção: para |z| grande, σ' ≈ 0 — o gradiente 'desaparece' ao ser propagado por várias camadas "
                           "(vanishing gradient). A ReLU resolve isso ao manter gradiente = 1 na zona ativa.",
                 "variant":"orange"},
            ]})

        # ── Passo 2: Forward Pass Época 1 ────────────────────────────────
        zA = X * W1
        hA = act(zA)
        zB = np.dot(hA, W2)
        hB = act(zB)
        zY = np.dot(hB, W3)
        yp = sigmoid(zY)   # saída sempre sigmoide
        err1 = 0.5 * (Y - yp) ** 2

        act_label = "\\text{ReLU}" if use_relu else "\\sigma"
        act_sym   = "ReLU" if use_relu else "σ"

        steps.append({"title":"Forward Pass — Época 1","sections":[
            {"type":"text","content":
             f"No forward pass, cada camada recebe os valores anteriores, faz uma combinação linear "
             f"e aplica {act_sym} (nas camadas ocultas) ou a Sigmoide (na saída). Veja cada conta."},
            {"type":"img","content": plot_forward(X,hA,hB,yp,W1,W2,W3,
                f"Época 1 — Forward Pass  (setas: pesos;  nós: ativações {act_sym}/σ)")},

            {"type":"subtitle","content":f"① Camada Oculta A  —  z_A = X · W1,  h_A = {act_sym}(z_A)"},
            {"type":"math","content":
             r"z_{A_1} = X \times W_1[0] = "
             + f"{X} \\times {W1[0]} = {zA[0]:.4f}"},
            {"type":"math","content":
             r"z_{A_2} = X \times W_1[1] = "
             + f"{X} \\times {W1[1]} = {zA[1]:.4f}"},
            {"type":"math","content":
             f"h_A = {act_sym}(z_A) = [{act_sym}({zA[0]:.4f}),\\ {act_sym}({zA[1]:.4f})] = [{hA[0]:.4f},\\ {hA[1]:.4f}]"},

            {"type":"subtitle","content":f"② Camada Oculta B  —  z_B = h_A · W2,  h_B = {act_sym}(z_B)"},
            {"type":"math","content":
             r"z_{B_1} = h_{A_1} \times W_2[0,0] + h_{A_2} \times W_2[1,0] = "
             + f"{hA[0]:.4f} \\times {W2[0,0]} + {hA[1]:.4f} \\times {W2[1,0]} = {zB[0]:.4f}"},
            {"type":"math","content":
             r"z_{B_2} = h_{A_1} \times W_2[0,1] + h_{A_2} \times W_2[1,1] = "
             + f"{hA[0]:.4f} \\times {W2[0,1]} + {hA[1]:.4f} \\times {W2[1,1]} = {zB[1]:.4f}"},
            {"type":"math","content":
             f"h_B = {act_sym}(z_B) = [{hB[0]:.4f},\\ {hB[1]:.4f}]"},

            {"type":"subtitle","content":"③ Saída  —  z_Y = h_B · W3,  ŷ = σ(z_Y)"},
            {"type":"math","content":
             r"z_Y = h_{B_1} \times W_3[0] + h_{B_2} \times W_3[1] = "
             + f"{hB[0]:.4f} \\times {W3[0]} + {hB[1]:.4f} \\times {W3[1]} = {zY:.4f}"},
            {"type":"math","content":
             r"\hat{y} = \sigma(z_Y) = \frac{1}{1+e^{-("
             + f"{zY:.4f}" + r")}} = " + f"{yp:.4f}"},
            {"type":"highlight",
             "content":f"Predição: ŷ = {yp:.4f}  |  Alvo: y = {Y}  |  Erro: E = {err1:.6f}",
             "variant":"orange"},
        ]})

        # ── Passo 3: Função de Custo ─────────────────────────────────────
        steps.append({"title":"Função de Custo — Época 1","sections":[
            {"type":"text","content":
             "Precisamos de um número que diga o quão errada está a rede. "
             "Usamos o Erro Quadrático Médio (MSE). O fator ½ é uma convenção matemática: "
             "ao derivar E para o backpropagation, o expoente 2 desce como fator e cancela o ½, "
             "deixando a expressão mais limpa."},
            {"type":"math","content": r"E = \frac{1}{2}(y - \hat{y})^2"},
            {"type":"subtitle","content":"Conta com os valores da Época 1"},
            {"type":"math","content":
             r"E = \frac{1}{2}(" + f"{Y} - {yp:.4f}" + r")^2"
             + r" = \frac{1}{2} \times (" + f"{Y-yp:.4f}" + r")^2"
             + r" = \frac{1}{2} \times " + f"{(Y-yp)**2:.6f}"
             + r" = " + f"{err1:.6f}"},
            {"type":"text","content":
             "Por que elevar ao quadrado? Duas razões: (1) faz erros positivos e negativos contribuírem igualmente; "
             "(2) penaliza erros grandes desproporcionalmente — 0.1² = 0.01, mas 0.5² = 0.25."},
            {"type":"highlight",
             "content":f"A rede prevê {yp:.4f}, mas deveria ser {Y}. "
                       "O backpropagation vai calcular em qual direção mover cada peso para reduzir esse valor.",
             "variant":"orange"},
        ]})

        # ── Passo 4: Backpropagation Época 1 ─────────────────────────────
        W1b=W1.copy(); W2b=W2.copy(); W3b=W3.copy()

        # Gradiente da saída (sempre sigmoide)
        dsY = d_sig(yp)
        dY  = (yp - Y) * dsY
        dW3 = dY * hB

        # Propaga para hB — derivada de act nas camadas ocultas usa z (pré-ativação)
        dsB  = dact_from_z(zB)
        dhB  = dY * W3 * dsB
        dW2  = np.outer(hA, dhB)

        dsA  = dact_from_z(zA)
        dhA  = np.dot(dhB, W2.T) * dsA
        dW1  = dhA * X

        W3 -= lr*dW3; W2 -= lr*dW2; W1 -= lr*dW1

        deriv_formula = (r"f'(z) = \begin{cases}0 & z\leq0\\1 & z>0\end{cases}"
                         if use_relu else
                         r"\sigma'(\hat{y}) = \hat{y}(1-\hat{y})")

        steps.append({"title":"Backpropagation — Época 1","sections":[
            {"type":"text","content":
             "O backpropagation aplica a regra da cadeia para calcular o gradiente do erro em relação a cada peso. "
             "Fazemos isso de trás para frente: começamos na saída e propagamos o sinal de erro para a entrada. "
             "Cada δ (delta) mede o quanto aquele neurônio é 'culpado' pelo erro final."},
            {"type":"img","content": plot_backprop(X,dW1,dW2,dW3,dhA,dhB,dY,
                f"Época 1 — Backpropagation  (setas: ∇W;  nós: δ de cada camada)")},

            {"type":"subtitle","content":"① Delta da Saída  δ_Y  (camada de saída usa Sigmoide)"},
            {"type":"math","content":
             r"\sigma'(\hat{y}) = \hat{y}\cdot(1-\hat{y}) = "
             + f"{yp:.4f} \\times {1-yp:.4f} = {dsY:.4f}"},
            {"type":"math","content":
             r"\delta_Y = (\hat{y} - y)\cdot\sigma'(\hat{y}) = "
             + f"({yp:.4f} - {Y}) \\times {dsY:.4f} = {dY:.6f}"},

            {"type":"subtitle","content":"② Gradiente de W3  (∇W3 = δ_Y · h_B)"},
            {"type":"math","content":
             r"\nabla W_3[0] = \delta_Y \times h_{B_1} = "
             + f"{dY:.4f} \\times {hB[0]:.4f} = {dW3[0]:.4f}"},
            {"type":"math","content":
             r"\nabla W_3[1] = \delta_Y \times h_{B_2} = "
             + f"{dY:.4f} \\times {hB[1]:.4f} = {dW3[1]:.4f}"},

            {"type":"subtitle","content":f"③ Delta de h_B  (derivada de {act_sym} aplicada em z_B)"},
            {"type":"math","content": deriv_formula
             + r"\quad \Rightarrow \quad f'(z_{B_1}) = " + f"{dsB[0]:.4f}"
             + r",\quad f'(z_{B_2}) = " + f"{dsB[1]:.4f}"},
            {"type":"math","content":
             r"\delta_{h_{B_1}} = \delta_Y \times W_3[0] \times f'(z_{B_1}) = "
             + f"{dY:.4f} \\times {W3b[0]:.4f} \\times {dsB[0]:.4f} = {dhB[0]:.6f}"},
            {"type":"math","content":
             r"\delta_{h_{B_2}} = \delta_Y \times W_3[1] \times f'(z_{B_2}) = "
             + f"{dY:.4f} \\times {W3b[1]:.4f} \\times {dsB[1]:.4f} = {dhB[1]:.6f}"},

            {"type":"subtitle","content":"④ Gradiente de W2"},
            {"type":"math","content":
             r"\nabla W_2 = \begin{bmatrix}"
             + f"{hA[0]:.3f}\\times{dhB[0]:.4f} & {hA[0]:.3f}\\times{dhB[1]:.4f}"
             + r"\\" + f"{hA[1]:.3f}\\times{dhB[0]:.4f} & {hA[1]:.3f}\\times{dhB[1]:.4f}"
             + r"\end{bmatrix} = \begin{bmatrix}"
             + f"{dW2[0,0]:.4f} & {dW2[0,1]:.4f}" + r"\\" + f"{dW2[1,0]:.4f} & {dW2[1,1]:.4f}"
             + r"\end{bmatrix}"},

            {"type":"subtitle","content":f"⑤ Delta de h_A e Gradiente de W1"},
            {"type":"math","content":
             r"f'(z_{A_1}) = " + f"{dsA[0]:.4f}" + r",\quad f'(z_{A_2}) = " + f"{dsA[1]:.4f}"},
            {"type":"math","content":
             r"\delta_{h_{A_1}} = (\delta_{h_{B_1}}\cdot W_2[0,0] + \delta_{h_{B_2}}\cdot W_2[0,1])\cdot f'(z_{A_1}) = "
             + f"({dhB[0]:.4f}\\cdot{W2b[0,0]} + {dhB[1]:.4f}\\cdot{W2b[0,1]})\\cdot{dsA[0]:.4f} = {dhA[0]:.6f}"},
            {"type":"math","content":
             r"\nabla W_1[0] = \delta_{h_{A_1}} \times X = "
             + f"{dhA[0]:.6f} \\times {X} = {dW1[0]:.6f}"},
            {"type":"math","content":
             r"\nabla W_1[1] = \delta_{h_{A_2}} \times X = "
             + f"{dhA[1]:.6f} \\times {X} = {dW1[1]:.6f}"},

            {"type":"subtitle","content":f"⑥ Atualização dos Pesos  (W ← W − {lr} · ∇W)"},
            {"type":"table","headers":["Peso","Antes","−lr·∇","Depois"],"rows":[
                ["W3[0]",   f"{W3b[0]:.4f}", f"−{lr}×{dW3[0]:.4f} = {-lr*dW3[0]:.4f}",   f"{W3[0]:.4f}"],
                ["W3[1]",   f"{W3b[1]:.4f}", f"−{lr}×{dW3[1]:.4f} = {-lr*dW3[1]:.4f}",   f"{W3[1]:.4f}"],
                ["W2[0,0]", f"{W2b[0,0]:.4f}", f"−{lr}×{dW2[0,0]:.4f} = {-lr*dW2[0,0]:.4f}", f"{W2[0,0]:.4f}"],
                ["W2[0,1]", f"{W2b[0,1]:.4f}", f"−{lr}×{dW2[0,1]:.4f} = {-lr*dW2[0,1]:.4f}", f"{W2[0,1]:.4f}"],
                ["W2[1,0]", f"{W2b[1,0]:.4f}", f"−{lr}×{dW2[1,0]:.4f} = {-lr*dW2[1,0]:.4f}", f"{W2[1,0]:.4f}"],
                ["W2[1,1]", f"{W2b[1,1]:.4f}", f"−{lr}×{dW2[1,1]:.4f} = {-lr*dW2[1,1]:.4f}", f"{W2[1,1]:.4f}"],
                ["W1[0]",   f"{W1b[0]:.4f}", f"−{lr}×{dW1[0]:.4f} = {-lr*dW1[0]:.4f}",   f"{W1[0]:.4f}"],
                ["W1[1]",   f"{W1b[1]:.4f}", f"−{lr}×{dW1[1]:.4f} = {-lr*dW1[1]:.4f}",   f"{W1[1]:.4f}"],
            ]},
        ]})

        # ── Passo 5: Época 2 ─────────────────────────────────────────────
        zA2=X*W1; hA2=act(zA2)
        zB2=np.dot(hA2,W2); hB2=act(zB2)
        zY2=np.dot(hB2,W3); yp2=sigmoid(zY2)
        err2=0.5*(Y-yp2)**2
        W1b2=W1.copy(); W2b2=W2.copy(); W3b2=W3.copy()
        dY2=(yp2-Y)*d_sig(yp2)
        dW3_2=dY2*hB2
        dhB2=dY2*W3*dact_from_z(zB2)
        dW2_2=np.outer(hA2,dhB2)
        dhA2=np.dot(dhB2,W2.T)*dact_from_z(zA2); dW1_2=dhA2*X
        W3-=lr*dW3_2; W2-=lr*dW2_2; W1-=lr*dW1_2

        steps.append({"title":"Época 2 — Ciclo Completo","sections":[
            {"type":"text","content":
             "Repetimos exatamente o mesmo ciclo com os pesos já modificados pela época 1. O erro deve ser menor."},
            {"type":"subtitle","content":"Forward Pass com pesos atualizados"},
            {"type":"img","content": plot_forward(X,hA2,hB2,yp2,W1,W2,W3,
                "Época 2 — Forward Pass (pesos ajustados pela época 1)")},
            {"type":"math","content":
             r"h_A = [" + f"{hA2[0]:.4f},\\ {hA2[1]:.4f}"
             + r"] \qquad h_B = [" + f"{hB2[0]:.4f},\\ {hB2[1]:.4f}" + r"]"},
            {"type":"math","content":
             r"\hat{y} = " + f"{yp2:.4f}"
             + r"\qquad E = " + f"{err2:.6f}"
             + r"\quad \bigl(\text{época 1: }" + f"{err1:.6f}" + r"\bigr)"},
            {"type":"highlight",
             "content":f"Redução do erro:  {err1:.6f} → {err2:.6f}  ({(1-err2/err1)*100:.1f}% menor)",
             "variant":"teal"},
            {"type":"subtitle","content":"Backpropagation — Época 2"},
            {"type":"img","content": plot_backprop(X,dW1_2,dW2_2,dW3_2,dhA2,dhB2,dY2,
                "Época 2 — Backpropagation")},
            {"type":"math","content":
             r"\delta_Y = " + f"{dY2:.6f}"
             + r"\quad \bigl(\text{época 1: }" + f"{dY:.6f}" + r"\bigr)"},
            {"type":"text","content":
             "Os gradientes ficaram menores, o que é esperado: "
             "quanto mais próximos do mínimo, menor o passo necessário."},
        ]})

        # ── Passo 6: Treinamento completo ────────────────────────────────
        hist_e=[err1,err2]; hist_p=[yp,yp2]
        for ep in range(3,epochs+1):
            zA_=X*W1; hA_=act(zA_)
            zB_=np.dot(hA_,W2); hB_=act(zB_)
            zY_=np.dot(hB_,W3); yp_=sigmoid(zY_)
            e_=0.5*(Y-yp_)**2
            hist_e.append(e_); hist_p.append(yp_)
            dY_=(yp_-Y)*d_sig(yp_)
            dW3_=dY_*hB_
            dhB_=dY_*W3*dact_from_z(zB_)
            dW2_=np.outer(hA_,dhB_)
            dhA_=np.dot(dhB_,W2.T)*dact_from_z(zA_); dW1_=dhA_*X
            W3-=lr*dW3_; W2-=lr*dW2_; W1-=lr*dW1_

        marks=sorted(set([max(0,x) for x in [0,9,epochs//4-1,epochs//2-1,epochs-1]]))
        rows=[[str(m+1),f"{hist_p[m]:.4f}",f"{hist_e[m]:.6f}"] for m in marks]

        steps.append({"title":f"Treinamento Completo — {epochs} Épocas","sections":[
            {"type":"text","content":
             f"As épocas 1 e 2 foram detalhadas nos passos anteriores. "
             f"Aqui executamos o mesmo ciclo para as épocas 3 a {epochs}."},
            {"type":"img","content": plot_learning_curve(hist_e,hist_p,Y,lr,X,epochs,act_name)},
            {"type":"subtitle","content":"Evolução ao longo do tempo"},
            {"type":"table","headers":["Época","Predição ŷ","Erro (MSE)"],"rows":rows},
            {"type":"highlight",
             "content":f"Redução total: {hist_e[0]:.6f} → {hist_e[-1]:.6f}  "
                       f"({(1-hist_e[-1]/hist_e[0])*100:.1f}% de melhora em {epochs} épocas)",
             "variant":"teal"},
        ]})

        # ── Passo 7: Estado Final ─────────────────────────────────────────
        zA_f=X*W1; hA_f=act(zA_f)
        zB_f=np.dot(hA_f,W2); hB_f=act(zB_f)
        zY_f=np.dot(hB_f,W3); yp_f=sigmoid(zY_f)

        steps.append({"title":"Estado Final da Rede","sections":[
            {"type":"text","content":
             f"Após {epochs} épocas de treinamento com {act_name} nas camadas ocultas, "
             f"veja como ficaram os pesos finais e o diagrama da rede."},
            {"type":"img","content": plot_forward(X,hA_f,hB_f,yp_f,W1,W2,W3,
                f"Após {epochs} épocas — Predição: {yp_f:.4f}  |  Alvo: {Y}")},
            {"type":"math","content":
             r"\hat{y}_{final} = " + f"{yp_f:.6f}"
             + r"\quad \longrightarrow \quad y = " + f"{Y}"
             + r"\quad \bigl(E_{final} = " + f"{hist_e[-1]:.8f}" + r"\bigr)"},
            {"type":"highlight",
             "content":f"Evolução:  ŷ = {hist_p[0]:.4f}  (época 1)  →  {yp_f:.4f}  (época {epochs})   alvo = {Y}",
             "variant":"teal"},
            {"type":"subtitle","content":"Pesos finais aprendidos"},
            {"type":"table","headers":["Peso","Valor final"],"rows":[
                ["W1[0]",   f"{W1[0]:.6f}"],
                ["W1[1]",   f"{W1[1]:.6f}"],
                ["W2[0,0]", f"{W2[0,0]:.6f}"], ["W2[0,1]", f"{W2[0,1]:.6f}"],
                ["W2[1,0]", f"{W2[1,0]:.6f}"], ["W2[1,1]", f"{W2[1,1]:.6f}"],
                ["W3[0]",   f"{W3[0]:.6f}"],
                ["W3[1]",   f"{W3[1]:.6f}"],
            ]},
        ]})

        summary = {
            "X": X, "Y": Y, "lr": lr, "epochs": epochs, "act": act_name,
            "w1_init": [float(w1_0),float(w1_1)],
            "w2_init": [[float(w2_00),float(w2_01)],[float(w2_10),float(w2_11)]],
            "w3_init": [float(w3_0),float(w3_1)],
            "yp_f": float(yp_f), "err_f": float(hist_e[-1])
        }

        return json.dumps({"ok":True,"steps":steps,"summary":summary})

    except Exception as e:
        import traceback
        return json.dumps({"ok":False,"error":str(e),"tb":traceback.format_exc()})
