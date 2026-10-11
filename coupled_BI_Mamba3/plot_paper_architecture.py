"""
Coupled-BI-Mamba3 论文风格模型图（矢量: SVG + PDF）
====================================================
四个子图（科研论文多联图排版）:
    (a) 整体架构（横向主流水线）
    (b) ISM 单模态块 (GLCE + BiMamba3 + BSSM 门控)
    (c) CoupledMamba3 融合层 (3×CrossMamba3Cell + GCMN 门控融合)
    (d) CrossMamba3Cell 展开 (以 tgt=Audio 为例, 严格跨模态)

维度/符号严格按最佳运行 (MOSI, d_model=128, 融合层 2, ISM 深度 2):
    ISM:    d_state 32, headdim 64 (4 heads), RoPE 50%, 融合 add/2
    Cell:   d_state 64, expand 2 → d_inner 256, headdim 32 (8 heads), ngroups 1
            Q/K ∈ (B,L,64), V/x ∈ (B,L,256), z ∈ (B,L,256), Δt/ΔA/trap ∈ (B,L,8)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle

plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

fig, ax = plt.subplots(figsize=(20, 14))
ax.set_xlim(0, 100)
ax.set_ylim(0, 70)
ax.axis('off')

ST = dict(
    T    = dict(fc='#E8F1FB', ec='#3178B6'),
    A    = dict(fc='#FDECEC', ec='#CC5A5A'),
    V    = dict(fc='#E8F5E9', ec='#4C9A57'),
    FUSE = dict(fc='#EFE9F7', ec='#7A56A8'),
    GATE = dict(fc='#FEF3E0', ec='#D08A1F'),
    SSM  = dict(fc='#E0F4F2', ec='#22867D'),
    HEAD = dict(fc='#EFEFF2', ec='#5F5F6B'),
    DARK = dict(fc='#2F3B4C', ec='#1C2733'),
    NOTE = dict(fc='#F8F8F6', ec='#A8A8A0'),
)
INK, SUB, PANEL = '#1F2430', '#6B7280', '#D9D9DE'
ARROW = '#3E4450'


def box(x, y, w, h, st, text='', fs=8.0, tc='#1F2430', bold=False, ls='-',
        lw=1.1, align='center', zorder=3, rs=0.5):
    b = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.08,rounding_size={rs}",
                       facecolor=st['fc'], edgecolor=st['ec'], linewidth=lw,
                       linestyle=ls, zorder=zorder)
    ax.add_patch(b)
    if text:
        tx = x + w / 2 if align == 'center' else x + 0.55
        ha = 'center' if align == 'center' else 'left'
        ax.text(tx, y + h / 2, text, ha=ha, va='center', fontsize=fs, color=tc,
                weight='bold' if bold else 'normal', zorder=zorder + 1, linespacing=1.5)
    return b


def panel(x, y, w, h, tag, title):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1,rounding_size=0.8",
                                facecolor='#FDFDFD', edgecolor=PANEL, linewidth=0.9, zorder=1))
    ax.text(x + w / 2, y + h - 1.05, f"({tag}) {title}",
            ha='center', va='center', fontsize=9.2, color=INK, weight='bold', zorder=6)


def arrow(x1, y1, x2, y2, color=ARROW, lw=1.25, ls='-', rad=0.0, zorder=2, ms=11):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='-|>', color=color, lw=lw, linestyle=ls,
                                shrinkA=1.0, shrinkB=1.0, mutation_scale=ms,
                                connectionstyle=f"arc3,rad={rad}"), zorder=zorder)


def line(x1, y1, x2, y2, color=ARROW, lw=1.25, ls='-'):
    ax.plot([x1, x2], [y1, y2], color=color, lw=lw, linestyle=ls, zorder=2,
            solid_capstyle='round')


def node(x, y, sym, r=0.68, ec='#3E4450', fs=7.5):
    ax.add_patch(Circle((x, y), r, facecolor='white', edgecolor=ec, lw=1.1, zorder=5))
    ax.text(x, y, sym, ha='center', va='center', fontsize=fs, color=INK,
            weight='bold', zorder=6)


def note(x, y, text, fs=6.4, color=SUB, ha='left', va='center', weight='normal'):
    ax.text(x, y, text, ha=ha, va=va, fontsize=fs, color=color, weight=weight,
            zorder=6, linespacing=1.5)


# ══════════════════════════════════════════════════════════════
# (a) 整体架构
# ══════════════════════════════════════════════════════════════
panel(1.5, 52.0, 97.0, 16.6, 'a', 'Coupled-BI-Mamba3 整体架构（MOSI）')
rT, rA, rV = 64.6, 60.1, 55.6
BH = 2.9

for rc, st, t1, t2 in ((rT, 'T', '文本 Token', '(B, 3, 50)'),
                       (rA, 'A', 'Data2Vec-Audio', '(B, 50, 768)'),
                       (rV, 'V', 'OpenFace3', '(B, 50, 18)')):
    box(2.6, rc - BH/2, 7.6, BH, ST[st], f'{t1}\n{t2}', fs=6.6)
box(12.4, rT - BH/2, 9.6, BH, ST['T'], 'RoBERTa-large 微调 (lr 1e-5)\nLinear 1024→128', fs=6.4)
box(12.4, rA - BH/2, 9.6, BH, ST['A'], 'Linear 768→128\n(离线预提取特征)', fs=6.4)
box(12.4, rV - BH/2, 9.6, BH, ST['V'], 'Linear 18→128\n(离线预提取特征)', fs=6.4)
for rc in (rT, rA, rV):
    arrow(10.3, rc, 12.3, rc)
    arrow(22.1, rc, 23.6, rc)
note(23.0, rV + 2.05, '(B, 50, 128)', fs=5.8, ha='center', color='#8A6D1C', weight='normal')

for rc in (rT, rA, rV):
    box(23.8, rc - BH/2, 7.2, BH, ST['SSM'], 'ISM × 2\n(见 b)', fs=7.0)
    arrow(31.1, rc, 32.6, rc)
note(31.85, rT + 2.15, '(B, 50, 128)×3', fs=5.8, ha='center', color='#8A6D1C', weight='normal')

box(32.8, 53.9, 9.8, 11.9, ST['FUSE'],
    'CoupledMamba3\n× 2 层\n每层 3 Cell\n+ GCMN 门控\n(见 c, d)', fs=6.9)

box(44.6, rA - BH/2, 6.8, BH, ST['FUSE'], 'Mean 池化\n(mask 感知)', fs=6.6)
arrow(42.7, rA, 44.5, rA)
box(53.4, rA - BH/2, 14.2, BH, ST['FUSE'],
    'stack [CTX; MAIN] → CoupledMamba3 ×2 (共享权重)\n→ 取 t = 1 (MAIN)', fs=6.3)
arrow(51.5, rA, 53.3, rA)
box(69.6, rA - BH/2, 9.6, BH, ST['HEAD'], 'Concat (B,384)\nLN → MLP → 1', fs=6.4)
arrow(67.7, rA, 69.5, rA)
box(81.2, rA - BH/2 - 0.4, 8.6, BH + 0.8, ST['DARK'],
    '情感分数\n$\\hat{y}\\,\\in\\,[−3, 3]$', fs=7.6, tc='white', bold=True)
arrow(79.3, rA, 81.1, rA)

box(53.4, 52.7, 14.2, 2.2, ST['NOTE'],
    'u_CTX = 前一句话语（虚线路径，共享全部编码器权重）', fs=6.2, ls='--', lw=1.0)
arrow(60.5, 54.95, 60.5, rA - BH/2 - 0.05, ls='--', lw=1.1, color=SUB)
note(2.9, 53.1, '单向上下文 (bidirectional = False) · dropout 0.6 · full-frame 方案B：池化对齐移至 ISM 之后', fs=6.0)

# ══════════════════════════════════════════════════════════════
# (b) ISM
# ══════════════════════════════════════════════════════════════
panel(1.5, 4.5, 29.5, 45.5, 'b', 'ISM 单模态块（T / A / V 各一套，独立权重）')
bx, bw = 6.2, 17.4
cx = bx + bw / 2

box(bx, 43.4, bw, 2.4, ST['HEAD'], '输入 $x\\,\\in\\,(B,\\,50,\\,128)$', fs=7.2)
box(bx, 40.2, bw, 2.4, ST['SSM'], '拼接 CLS + 可学习位置编码 $E_{pos}$', fs=7.0)
note(24.2, 41.4, '(B, 51, 128)', fs=5.8, color='#8A6D1C', weight='normal')
arrow(cx, 43.35, cx, 42.75)

ax.add_patch(FancyBboxPatch((bx - 0.7, 19.4), bw + 1.4, 19.9,
                            boxstyle="round,pad=0.08,rounding_size=0.7",
                            facecolor='#F3FBFA', edgecolor=ST['SSM']['ec'],
                            lw=1.0, linestyle=(0, (4, 2)), zorder=2))
note(bx - 0.45, 38.78, 'Block × 2（深度 2）', fs=6.8, ha='left', color='#22867D', weight='bold')

box(bx, 35.8, bw, 2.3, ST['SSM'], 'RMSNorm（融合残差, 双流）', fs=7.0)
arrow(cx, 40.15, cx, 38.35)
box(bx, 31.9, bw, 3.0, ST['SSM'],
    'GLCE：全局 $\\mathbf{W}_g x$ + 局部 Conv1D$_{k=3}$ + 短路\n→ RMSNorm', fs=6.7)
arrow(cx, 35.75, cx, 35.0)
box(bx, 27.0, bw, 3.6, ST['SSM'],
    'BiMamba3（双向 Mamba-3）\n前向扫描 →  ⊕  ← 反向扫描\n$y = \\frac{1}{2}(y_{fwd} + flip(y_{bwd}))$', fs=6.7)
arrow(cx, 31.85, cx, 30.7)
box(bx, 24.4, bw, 2.2, ST['GATE'], 'BSSM 门控  $G=\\mathrm{SiLU}(\\mathrm{MLP}(h))$', fs=7.0)
arrow(cx, 26.95, cx, 26.72)
node(cx, 23.2, '⊙')
note(cx + 1.0, 23.2, '逐元素门控', fs=6.0, ha='left')
arrow(cx, 22.48, cx, 20.6)

# 残差流（左弧线）
line(cx - bw/2, 44.6, 3.6, 44.6, lw=1.1)
line(3.6, 44.6, 3.6, 17.6, lw=1.1)
node(3.6, 17.6, '⊕')
arrow(3.6, 17.6, cx - bw/2 - 0.1, 16.4, lw=1.1, rad=0.25)
note(2.4, 31.0, '双流残差\n(hidden,\nresidual)', fs=6.0, ha='center')

box(bx, 14.6, bw, 2.4, ST['SSM'], '输出序列 $h$（去 CLS）  (B, 50, 128)', fs=6.9)
arrow(cx, 19.35, cx, 17.15)
box(bx + 1.2, 10.9, bw - 2.4, 2.2, ST['NOTE'],
    'CLS 向量 (B,128) → sub_fc（MOSI 未激活）', fs=6.2, ls='--', lw=1.0)
arrow(cx, 14.55, cx, 13.25, ls='--', lw=1.0, color=SUB)
note(24.6, 33.0, 'd_state 32\nheaddim 64 (4 heads)\nRoPE 50% · 融合 add/2\npad 位逐层 zero-out', fs=6.2)

# ══════════════════════════════════════════════════════════════
# (c) CoupledMamba3 融合层
# ══════════════════════════════════════════════════════════════
panel(32.5, 4.5, 31.5, 45.5, 'c', 'CoupledMamba3 融合层（× 2, ℓ = 1, 2）')
ccx = 48.25

chip_x = [34.2, 44.1, 54.0]
chip_st = ['A', 'V', 'T']
chip_lab = ['$u_A$', '$u_V$', '$u_T$']
for x0, st, lab in zip(chip_x, chip_st, chip_lab):
    box(x0, 43.0, 7.6, 2.2, ST[st], f'{lab}  (B,L,128)', fs=6.8)
    arrow(x0 + 3.8, 42.95, x0 + 3.8, 41.4)
note(63.4, 46.0, 'ISM 后特征 (对齐至 L=50)', fs=5.9, ha='right')

for x0, tgt in zip(chip_x, ('tgt = Audio', 'tgt = Visual', 'tgt = Text')):
    box(x0, 37.2, 7.6, 4.1, ST['FUSE'], f'CrossMamba3Cell\n{tgt}\n(展开见 d)', fs=6.3)

# Cell 输出 → ⊕残差 → LN
for x0 in chip_x:
    xc = x0 + 3.8
    arrow(xc, 37.15, xc, 35.75)
    node(xc, 34.95, '⊕')
    arrow(xc, 34.22, xc, 34.0)
    box(xc - 1.7, 32.0, 3.4, 1.9, ST['HEAD'], 'LN', fs=6.8)
    # 残差旁路（chip 右侧绕行）
    line(x0 + 7.6, 44.1, x0 + 8.7, 44.1, color=SUB, lw=0.9)
    line(x0 + 8.7, 44.1, x0 + 8.7, 34.95, color=SUB, lw=0.9)
    arrow(x0 + 8.7, 34.95, xc + 0.72, 34.95, color=SUB, lw=0.9)
note(61.9, 42.15, '残差 ⊕ + LN', fs=5.6, ha='right')

# GCMN 容器
ax.add_patch(FancyBboxPatch((33.4, 12.9), 29.8, 16.7,
                            boxstyle="round,pad=0.08,rounding_size=0.7",
                            facecolor='#FFFDF6', edgecolor=ST['GATE']['ec'],
                            lw=1.0, linestyle=(0, (4, 2)), zorder=2))
note(ccx, 29.15, 'GCMN 三流门控融合（每层末端）', fs=7.0, ha='center', color='#B06E0F', weight='bold')

# cross 路径: LN 输出汇入 F_cross
for x0 in chip_x:
    line(x0 + 3.8, 31.95, x0 + 3.8, 30.7, lw=1.0)
line(38.0, 30.7, 57.8, 30.7, lw=1.0)
arrow(55.8, 30.7, 55.8, 28.75, lw=1.0)

box(35.0, 24.0, 12.4, 4.6, ST['GATE'],
    '单模态保留 $F_{uni}$\nUnimodalBSSM(raw)\nLN → MLP(↑2) → SiLU → MLP + 残差', fs=5.9)
box(49.6, 24.0, 12.4, 4.6, ST['GATE'],
    '跨模态增强 $F_{cross}$\nCrossModalBSSM\ncat (B,L,384) → LN → MLP + 残差\n→ 投影 ×3 (384→128)', fs=5.9)
# raw 路径（虚线, 从 chips 左侧绕入 F_uni）
line(34.2, 44.1, 33.7, 44.1, color=SUB, lw=0.9, ls='--')
line(33.7, 44.1, 33.7, 26.3, color=SUB, lw=0.9, ls='--')
arrow(33.7, 26.3, 34.95, 26.3, color=SUB, lw=0.9, ls='--')
ax.text(33.2, 35.2, 'raw = 本层输入 (ISM 后, 虚线)', rotation=90, ha='center', va='center',
        fontsize=5.8, color=SUB, zorder=6)

box(38.4, 19.6, 23.0, 2.6, ST['GATE'],
    '门控 $G=\\sigma(\\mathrm{MLP}(\\mathrm{cat}\\;F_{cross}))$ → $g_L, g_A, g_V$', fs=6.6)
arrow(55.8, 23.95, 55.8, 22.35)
node(48.0, 18.3, '⊙')
arrow(48.0, 19.55, 48.0, 19.05)
# F_uni → ⊕（沿 GCMN 容器左内侧走线）
line(36.9, 23.95, 36.9, 18.3, lw=1.0)
arrow(36.9, 18.3, 47.27, 18.3, lw=1.0)
box(37.6, 14.9, 21.4, 2.4, ST['FUSE'],
    '$out_m=\\mathrm{LN}(F_{uni,m}+g_m\\,\\odot\\,F_{cross,m})$   (B,L,128)×3', fs=6.4)
arrow(48.0, 17.58, 48.0, 17.42, lw=1.0)
arrow(48.3, 14.85, ccx, 13.6)
box(37.2, 10.2, 22.0, 2.3, ST['FUSE'], '层输出 (B,L,128) × 3 → 下一层 / 共堆叠 2 次', fs=6.6)

# ══════════════════════════════════════════════════════════════
# (d) CrossMamba3Cell
# ══════════════════════════════════════════════════════════════
panel(65.5, 4.5, 33.0, 45.5, 'd', 'CrossMamba3Cell（tgt = Audio 为例, 严格跨模态）')

box(67.2, 43.2, 6.6, 2.2, ST['A'], '$u_A$ (目标)', fs=7.0)
box(75.4, 43.2, 6.6, 2.2, ST['V'], '$u_V$ (源)', fs=7.0)
box(83.6, 43.2, 6.6, 2.2, ST['T'], '$u_T$ (源)', fs=7.0)
note(96.9, 46.0, '(B, L, 128)', fs=5.9, ha='right')

box(70.0, 38.9, 22.2, 2.9, ST['GATE'],
    'weight_net：MLP(cat[$u_A$,$u_V$,$u_T$] 384→128→2)\n→ softmax → ($w_V$, $w_T$)', fs=6.4)
for cx0 in (70.5, 78.7, 86.9):
    arrow(cx0, 43.15, cx0, 41.95)

# u_A 直达目标路径（不经过 weight_net）
line(68.4, 43.15, 68.4, 37.5, lw=1.0)
arrow(68.4, 37.5, 68.4, 37.15, lw=1.0)

box(67.0, 31.4, 14.0, 5.6, ST['A'],
    '目标路径 in_proj_tgt($u_A$)\n$z$ 门控 (B,L,256)\n$\\Delta t,\\Delta A$,trap (B,L,8), angles(16)\n$C=c_{proj}(u_A)=Q$ (B,L,64)', fs=6.0)
box(82.4, 31.4, 14.6, 5.6, ST['V'],
    '源加权（K/V 全部来自源）\n$K=w_V K_V + w_T K_T$ (B,L,64)\n$x=w_V V_V + w_T V_T$ (B,L,256)\n$K_m=b_{proj}(u_m),\\; V_m=v_{proj}(u_m)$', fs=6.0)
arrow(86.9, 38.85, 86.9, 37.15)
note(88.3, 38.2, '⊗ $w_m$ 加权', fs=5.9)

box(67.0, 28.0, 30.0, 2.4, ST['SSM'], 'RMSNormGated(Q) · RMSNormGated(K)   (ngroups = 1)', fs=6.6)
arrow(74.0, 31.35, 74.0, 30.55)
arrow(89.7, 31.35, 89.7, 30.55)

box(67.0, 20.6, 30.0, 6.4, ST['DARK'],
    'Mamba-3 SISO 选择性扫描核\n'
    '$A=-\\mathrm{softplus}(\\Delta A)$, 截断 $\\leq -A_{floor}$；$\\Delta=\\mathrm{softplus}(\\Delta t+b)$\n'
    'ADT = $A\\cdot\\Delta t$（梯形离散化）· RoPE(50% 维度, 作用于 Q/K)\n'
    '$y_t=C_t\\,S_t+D\\odot x_t$,  $S_t=e^{ADT_t}S_{t-1}+K_t^{\\top}V_t$', fs=6.4, tc='white')
arrow(82.0, 27.95, 82.0, 27.15)

box(73.4, 16.4, 12.6, 2.4, ST['FUSE'], 'out_proj 256→128', fs=6.8)
arrow(82.0, 20.55, 82.0, 18.95)
# z 门控（左侧走线）
line(67.0, 35.4, 66.35, 35.4, color=SUB, lw=0.9)
line(66.35, 35.4, 66.35, 17.6, color=SUB, lw=0.9)
node(71.6, 17.6, '⊙')
arrow(66.35, 17.6, 70.87, 17.6, color=SUB, lw=0.9)
note(67.0, 19.6, '$z$ 门控 (SiLU)', fs=5.8, ha='left', color=SUB)
arrow(71.6 + 0.72, 17.6, 73.35, 17.6, lw=1.0)

# 残差旁路 u_A（最左走线）
line(67.2, 44.3, 65.85, 44.3, color=SUB, lw=0.9, ls='--')
line(65.85, 44.3, 65.85, 12.6, color=SUB, lw=0.9, ls='--')
node(79.7, 12.6, '⊕')
line(79.7, 16.35, 79.7, 13.35, lw=1.0)
arrow(65.85, 12.6, 78.97, 12.6, color=SUB, lw=0.9, ls='--')
note(67.4, 14.6, '$u_A$ 残差（虚线）', fs=5.8, ha='left', color=SUB)

box(82.0, 11.4, 4.4, 2.4, ST['HEAD'], 'LN', fs=7.0)
arrow(79.7 + 0.72, 12.6, 81.95, 12.6)
box(87.8, 11.4, 9.6, 2.4, ST['FUSE'], 'out$_A$\n(B,L,128)', fs=6.4)
arrow(86.45, 12.6, 87.75, 12.6)
note(96.9, 7.6, '每层 3 个 Cell\n(tgt ∈ {A,V,T})\n输出经 GCMN 融合 (见 c)', fs=6.0, ha='right')

# ══════════════════════════════════════════════════════════════
# 底部注脚
# ══════════════════════════════════════════════════════════════
note(50.0, 2.6,
     'Coupled-BI-Mamba3 · MOSI 最佳运行（seed 2024）：test Acc2 89.79 / Acc7 52.77 / F1 89.77 / MAE 0.589 · '
     'd_model=128 · 融合层 2 · ISM 深度 2 · dropout 0.6',
     fs=7.0, ha='center', color='#4A4A55', weight='bold')
note(50.0, 1.2,
     '图例: 蓝=文本  红=音频  绿=视觉  紫=CoupledMamba3 跨模态融合  青=SSM/ISM  橙=门控(GCMN/BSSM/源权重)  灰=投影与输出头；'
     '实线=主话语路径, 虚线=上下文/残差/未激活路径',
     fs=6.2, ha='center', color=SUB)

fig.subplots_adjust(left=0.004, right=0.996, top=0.996, bottom=0.004)
fig.savefig('coupled_bimamba3_paper.svg', facecolor='white')
fig.savefig('coupled_bimamba3_paper.pdf', facecolor='white')
fig.savefig('coupled_bimamba3_paper.png', dpi=260, facecolor='white')
print('saved: coupled_bimamba3_paper.svg / .pdf / .png')
