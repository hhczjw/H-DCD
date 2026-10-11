"""
Coupled-BI-Mamba3 最佳运行 (MOSI, dt_bs4_sd2024_do0.6_lr5e-4_ds64_id2_sl0.1)
模型拓扑图生成器 — 科研汇报用 (横版, 300 DPI PNG + PDF + SVG)

拓扑严格对应代码路径:
    train.py -> MSAClassifier (models/classifier.py)
      ├─ TextPretrainedEncoder  RoBERTa-large 微调 (models/text_encoder.py)
      ├─ ISMEncoder ×2          GLCE + BiMamba3 + BSSM门控 (layers/ism.py)
      ├─ CoupledMamba3Fork ×2   CrossMamba3Cell + GCMN (models/coupled_mamba3_fork.py,
      │                         models/gcmn_fusion.py), ism_full_frame=方案B
      ├─ encode_with_context    对话级 [CTX; MAIN] 融合, 单向, 权重共享
      │                         (models/context_fusion.py, bidirectional=False)
      └─ Head: LN(384) → 128 → GELU → Dropout(0.6) → 1
    Loss: MultiTaskLoss( M: SmoothL1 β=0.5 )  (MOSI 仅主任务激活)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['svg.fonttype'] = 'none'

fig, ax = plt.subplots(figsize=(20, 12.6))
ax.set_xlim(0, 100)
ax.set_ylim(0, 63)
ax.axis('off')

# ─── 配色: 浅底 + 深边框 + 深色文字 (投影可读) ───
ST = dict(
    T    = dict(fc='#DCEBFB', ec='#2E6DB4'),   # 文本 蓝
    A    = dict(fc='#FDE3E8', ec='#C94F6D'),   # 音频 红
    V    = dict(fc='#DFF2E1', ec='#3F9E52'),   # 视频 绿
    FUSE = dict(fc='#EADDF5', ec='#7C4DA4'),   # 跨模态融合 紫
    GATE = dict(fc='#FDEBD0', ec='#D98A1F'),   # 门控 橙
    SSM  = dict(fc='#D6F2EF', ec='#1F9E93'),   # SSM / Mamba 青
    HEAD = dict(fc='#ECECEF', ec='#5A5A66'),   # 输出头 灰
    DARK = dict(fc='#34495E', ec='#1F2D3D'),   # 深色(输出/损失)
    NOTE = dict(fc='#F7F7F5', ec='#9A9A92'),   # 注释 浅灰
)
ARROW = '#4A4A52'
SUB = '#8A8A92'


def box(x, y, w, h, st, text='', fs=9.2, tc='#22222A', bold=False, ls='-',
        lw=1.4, align='center', zorder=3, radius=0.55):
    b = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad=0.10,rounding_size={radius}",
                       facecolor=st['fc'], edgecolor=st['ec'], linewidth=lw,
                       linestyle=ls, zorder=zorder)
    ax.add_patch(b)
    if text:
        tx = x + w / 2 if align == 'center' else x + 0.7
        ha = 'center' if align == 'center' else 'left'
        ax.text(tx, y + h / 2, text, ha=ha, va='center', fontsize=fs,
                color=tc, weight='bold' if bold else 'normal', zorder=zorder + 1,
                linespacing=1.45)
    return b


def arrow(x1, y1, x2, y2, color=ARROW, lw=1.7, ls='-', rad=0.0, zorder=2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='-|>', color=color, lw=lw,
                                linestyle=ls, shrinkA=1.5, shrinkB=1.5,
                                connectionstyle=f"arc3,rad={rad}",
                                mutation_scale=15), zorder=zorder)


def line(x1, y1, x2, y2, color=ARROW, lw=1.7, ls='-'):
    ax.plot([x1, x2], [y1, y2], color=color, lw=lw, linestyle=ls,
            solid_capstyle='round', zorder=2)


def caption(x, y, text, fs=10.5, color='#3B3B46', ha='center', bold=True):
    ax.text(x, y, text, ha=ha, va='center', fontsize=fs, color=color,
            weight='bold' if bold else 'normal', zorder=5)


# ══════════════════════════════ 标题 ══════════════════════════════
caption(50, 61.7, 'Coupled-BI-Mamba3 总体拓扑（MOSI 最佳配置：test Acc2 89.79 / Acc7 52.77）', fs=15)
caption(50, 60.0, 'd_model=128 · 融合层数 2 · ISM 深度 2 · ism_full_frame（方案B）· dropout 0.6 · seed 2024',
        fs=9.5, color=SUB, bold=False)

# ══════════════════════════ 主干流程（左侧） ══════════════════════════
xT, xA, xV, W = 12.0, 29.0, 46.0, 13.2

# ── ① 输入与投影 ──
caption(30, 58.7, '① 多模态输入与编码投影', fs=10.5)
box(xT - W/2, 55.2, W, 2.6, ST['T'], '文本话语 Token 序列\n(B, 3, 50)  [ids/mask/seg]', fs=8.6)
box(xA - W/2, 53.4, W, 4.4, ST['A'], 'Data2Vec-Audio 特征\n(B, 50, 768)\n（离线预提取）', fs=8.6)
box(xV - W/2, 53.4, W, 4.4, ST['V'], 'OpenFace3 视觉特征\n(B, 50, 18)\n（离线预提取）', fs=8.6)

box(xT - W/2, 50.6, W, 3.2, ST['T'], 'RoBERTa-large 微调\n(lr 1e-5) → (B, 50, 1024)', fs=8.6)
arrow(xT, 55.2, xT, 53.85)
arrow(xT, 50.6, xT, 49.85)
arrow(xA, 53.4, xA, 49.85)
arrow(xV, 53.4, xV, 49.85)

box(xT - W/2, 47.3, W, 2.5, ST['T'], 'Linear 1024 → 128', fs=8.6)
box(xA - W/2, 47.3, W, 2.5, ST['A'], 'Linear 768 → 128', fs=8.6)
box(xV - W/2, 47.3, W, 2.5, ST['V'], 'Linear 18 → 128', fs=8.6)
for xc in (xT, xA, xV):
    arrow(xc, 47.3, xc, 46.45)

# ── ② ISM ──
for xc, lab in ((xT, 'ISM-Text'), (xA, 'ISM-Audio'), (xV, 'ISM-Video')):
    box(xc - W/2, 38.3, W, 8.1, ST['SSM'])
    caption(xc - W/2 + 0.6, 45.7, f'② {lab} ×2（模态独立权重）', fs=8.4, ha='left')
    box(xc - W/2 + 0.9, 43.0, W - 1.8, 1.9, ST['SSM'], 'CLS ⊕ 可学习位置编码', fs=8.0)
    box(xc - W/2 + 0.9, 39.1, W - 1.8, 3.0, ST['SSM'], '[ GLCE → BiMamba3 →\nBSSM 门控 ] × 2', fs=8.2)
    arrow(xc, 43.0, xc, 42.15)
caption(37.7, 37.6, '方案B：池化对齐后移\n（L=50 恒等）', fs=7.0, color=SUB, bold=False)

# sub 辅助头（结构存在, MOSI 上不参与损失）
box(52.9, 40.2, 6.4, 3.4, ST['NOTE'], 'sub_fc_T/A/V\n(128→1)\nMOSI：未参与损失', fs=7.4, ls='--', lw=1.1)
arrow(xV + W/2, 41.6, 52.85, 42.0, ls='--', lw=1.1, color=SUB)
caption(56.1, 44.4, 'ISM CLS', fs=7.6, color=SUB, bold=False)

# ISM → 融合容器
for xc in (xT, xA, xV):
    arrow(xc, 38.3, xc, 36.95)

# ── ③ CoupledMamba3Fork ×2 ──
box(3.6, 26.8, 54.4, 9.6, ST['FUSE'], '')
caption(30.8, 35.7, '③ CoupledMamba3Fork × 2 层（严格跨模态 Q/K/V 融合，ℓ = 1, 2）', fs=10)
line(13.2, 36.95, 46.8, 36.95, lw=1.7)
for xc in (xT, xA, xV):
    line(xc, 36.95, xc, 36.95)
cell_x = [5.4, 22.2, 39.0]
cw = 15.6
for xc, lab in zip(cell_x, ('CrossMamba3Cell\ntgt=Audio（Q←A）',
                            'CrossMamba3Cell\ntgt=Visual（Q←V）',
                            'CrossMamba3Cell\ntgt=Text（Q←T）')):
    box(xc, 31.2, cw, 3.4, ST['FUSE'], lab, fs=8.2)
    arrow(xc + cw/2, 36.95, xc + cw/2, 34.68)
box(8.0, 28.0, 45.6, 2.4, ST['GATE'],
    'GCMN 三流门控：F = F_uni（UnimodalBSSM，本层输入）+ σ(G) ⊙ F_cross（CrossModalBSSM）→ LN',
    fs=7.8)
for xc in cell_x:
    arrow(xc + cw/2, 31.2, xc + cw/2, 30.48)

# GCMN → 池化
arrow(13.2, 28.0, 8.0, 22.58, rad=0.08)
arrow(30.0, 28.0, 11.8, 22.58, rad=0.06)
arrow(46.8, 28.0, 15.6, 22.58, rad=0.06)

# ── ④ 对话级上下文融合 ──
caption(53.4, 24.7, '④ 对话级上下文融合（单向 CTX→MAIN）', fs=9.6, ha='right')
box(4.0, 19.3, 15.6, 3.2, ST['FUSE'], 'Mean 池化（mask 感知）\n→ (B,128) × 3 模态', fs=8.2)
box(21.6, 19.3, 15.6, 3.2, ST['FUSE'], 'stack [ u_CTX ; u_MAIN ]\n→ (B, 2, 128) × 3', fs=8.2)
arrow(19.65, 20.9, 21.55, 20.9)
box(21.6, 15.9, 15.6, 2.6, ST['NOTE'], '上下文话语 u_CTX（前一句）\n共享全部编码权重', fs=8.0, ls='--', lw=1.1)
arrow(29.4, 18.55, 29.4, 19.25, ls='--', lw=1.2, color=SUB)
box(39.2, 19.3, 14.4, 3.2, ST['FUSE'], 'CoupledMamba3Fork ×2\n（与话语级共享权重）', fs=8.2)
arrow(37.25, 20.9, 39.15, 20.9)
box(39.2, 15.9, 14.4, 2.6, ST['FUSE'], '取 t = 1（MAIN）\n→ (B,128) × 3', fs=8.2)
arrow(46.4, 19.25, 46.4, 18.55)

# ── ⑤ 输出头 ──
caption(3.8, 14.6, '⑤ 回归输出头', fs=10, ha='left')
box(4.0, 10.6, 13.2, 3.0, ST['HEAD'], 'Concat [T;A;V]\n(B, 384)', fs=8.4)
box(18.4, 10.6, 10.2, 3.0, ST['HEAD'], 'LayerNorm', fs=8.4)
box(29.8, 10.6, 14.6, 3.0, ST['HEAD'], 'Linear 384→128\nGELU · Dropout 0.6', fs=8.4)
box(45.6, 10.6, 12.0, 3.0, ST['HEAD'], 'Linear 128→1', fs=8.4)
arrow(46.4, 15.85, 46.4, 14.25)
line(46.4, 14.25, 10.6, 14.25)
arrow(10.6, 14.25, 10.6, 13.72)
arrow(17.25, 12.1, 18.35, 12.1)
arrow(28.65, 12.1, 29.75, 12.1)
arrow(44.45, 12.1, 45.55, 12.1)

box(24.0, 6.7, 16.0, 2.9, ST['DARK'], '情感分数 $\\hat{y}$ ∈ [−3, 3]', fs=10.5, tc='white', bold=True)
arrow(51.6, 10.55, 51.6, 8.15)
arrow(51.55, 8.15, 40.15, 8.15)

# ── 训练横幅 ──
box(3.6, 1.6, 54.4, 3.7, ST['DARK'],
    '损失：MultiTaskLoss = SmoothL1(β=0.5)·(主任务 M)（MOSI 标签仅 M，T/A/V 子头未激活）\n'
    '优化：AdamW 双学习率组（RoBERTa 1e-5 / 其余 5e-4）· wd 1e-4 · warmup 15% + 余弦衰减 · batch 4 · 梯度裁剪 0.1 · ≤120 轮，早停 30',
    fs=8.4, tc='white')
arrow(32.0, 6.65, 32.0, 5.38)

# ══════════════════════════ 右侧放大面板 ══════════════════════════
RX, RW = 61.0, 38.0

# ── 放大①: ISM Block ──
box(RX, 43.0, RW, 16.4, ST['NOTE'], '', lw=1.2)
caption(RX + RW/2, 58.6, '放大①：ISM 单模态块（T / A / V 各一套，深度 2）', fs=9.6)
bx, bw = RX + 2.4, 23.6
items = [
    ('输入 x ∈ (B, 50, 128)', 'HEAD', 1.5),
    ('CLS ⊕ 可学习位置编码', 'SSM', 1.5),
    ('RMSNorm（融合残差，双流）', 'SSM', 1.5),
    ('GLCE：全局 Linear + 局部 Conv1d(k=3) + 短路 → RMSNorm', 'SSM', 2.4),
    ('BiMamba3 双向 Mamba-3（前向⊕后向）\nd_state 32 · headdim 64 · RoPE 50% · 融合 add/2', 'SSM', 2.8),
    ('BSSM 门控：SiLU(MLP(X)) ⊙ X', 'GATE', 1.7),
]
y = 57.6
centers = []
items_h = [1.4, 1.4, 1.4, 2.3, 2.7, 1.6]
for (txt, key, _), h in zip(items, items_h):
    box(bx, y - h, bw, h, ST[key], txt, fs=7.4)
    centers.append((bx + bw/2, y, y - h))
    y -= h + 0.35
for i in range(len(centers) - 1):
    arrow(centers[i][0], centers[i][2], centers[i+1][0], centers[i+1][1])
arrow(bx + bw + 1.3, 56.5, bx + bw + 1.3, 47.4, rad=-0.5, lw=1.2, color=SUB)
caption(bx + bw + 2.7, 51.8, '残差流\nhidden /\nresidual\n跨层传递', fs=7.6, color=SUB, bold=False)
caption(bx + bw + 2.7, 45.6, '× 2 层', fs=8.6, color='#7C4DA4')
caption(RX + 1.2, 43.95,
        '输出：序列 (去 CLS) 供跨模态融合；CLS → sub_fc（虚线，MOSI 未激活）· pad 位逐层 zero-out',
        fs=7.8, color=SUB, bold=False, ha='left')

# ── 放大②: CrossMamba3Cell ──
box(RX, 13.2, RW, 28.4, ST['NOTE'], '', lw=1.2)
caption(RX + RW/2, 40.7, '放大②：CrossMamba3Cell（以目标模态 Audio 为例，严格跨模态 v_self = 0）', fs=9.4)
chip_w, chip_h = 9.6, 2.0
box(RX + 1.6, 36.6, chip_w, chip_h, ST['A'], 'u_A（目标）', fs=7.8)
box(RX + 13.4, 36.6, chip_w, chip_h, ST['V'], 'u_V（源）', fs=7.8)
box(RX + 25.2, 36.6, chip_w, chip_h, ST['T'], 'u_T（源）', fs=7.8)
box(RX + 1.6, 32.7, 33.2, 2.4, ST['GATE'],
    'weight_net：MLP(concat u_A, u_V, u_T) → softmax → (w_V, w_T)', fs=7.6)
for cx in (RX + 1.6 + chip_w/2, RX + 13.4 + chip_w/2, RX + 25.2 + chip_w/2):
    arrow(cx, 36.55, cx, 35.18)
box(RX + 1.6, 27.7, 15.8, 4.1, ST['A'],
    'in_proj_tgt(u_A)：\nz 门控 · Δt / ΔA / trap / angles\nC = Q（查询来自目标）', fs=7.2)
box(RX + 19.0, 27.7, 15.8, 4.1, ST['V'],
    '源加权（键 / 值全部来自源）：\nB = K = w_V·K_V + w_T·K_T\nV = x = w_V·V_V + w_T·V_T', fs=7.2)
arrow(RX + 9.5, 32.65, RX + 9.5, 31.88)
arrow(RX + 26.9, 32.65, RX + 26.9, 31.88)
box(RX + 1.6, 22.5, 33.2, 4.1, ST['SSM'],
    'Mamba-3 SISO 选择性扫描核\nA = −softplus(·) + A_floor · 梯形离散化 Δ · RoPE(50%)\nRMSNormGated(Q, K) · D 跳连 · z 门控', fs=7.2)
arrow(RX + 9.5, 27.65, RX + 9.5, 26.68)
arrow(RX + 26.9, 27.65, RX + 26.9, 26.68)
box(RX + 1.6, 18.7, 15.8, 2.6, ST['FUSE'], 'out_proj → d_model', fs=7.6)
box(RX + 19.0, 18.7, 15.8, 2.6, ST['FUSE'], '⊕ 残差(u_A) → LayerNorm', fs=7.6)
arrow(RX + 9.5, 22.45, RX + 9.5, 21.38)
arrow(RX + 26.9, 22.45, RX + 26.9, 21.38)
arrow(RX + 17.48, 20.0, RX + 18.92, 20.0)
caption(RX + RW/2, 16.8, '每层 3 个 Cell（tgt = A / V / T 各一）+ GCMN 门控融合；融合层共堆叠 2 次',
        fs=8.0, color=SUB, bold=False)

# ── 放大③: 配置与结果 ──
box(RX, 1.6, RW, 10.2, ST['HEAD'], '', lw=1.2)
caption(RX + RW/2, 10.9, '最佳运行：logs/MSA_MOSI_20260607_045751.log', fs=9.4)
ax.text(RX + 1.0, 8.7,
        'MOSI test（best-Acc2 ckpt）：Acc2 = 89.79   Acc7 = 52.77   F1 = 89.77\n'
        'MAE = 0.589   Corr = 0.871   Acc5 = 61.37   （seed 2024 · 第 70 轮早停）\n'
        '关键超参：d_model 128 · 融合层数 2 · d_state 64 · ISM depth 2 (d_state 32) · dropout 0.6\n'
        '数据：MOSI 训练集 1284 样本 · L = 50 · unaligned · mean 池化',
        ha='left', va='center', fontsize=8.2, color='#33333C', linespacing=1.75)

# ══════════════════════════ 图例 ══════════════════════════
leg = [('文本', 'T'), ('音频', 'A'), ('视觉', 'V'), ('跨模态融合', 'FUSE'),
       ('门控', 'GATE'), ('SSM/Mamba', 'SSM'), ('输出头', 'HEAD')]
lx = 3.8
for name, key in leg:
    box(lx, 0.2, 1.5, 0.9, ST[key], '', radius=0.3)
    ax.text(lx + 1.9, 0.65, name, ha='left', va='center', fontsize=8.0, color='#3B3B46')
    lx += 1.9 + len(name) * 1.05 + 1.5

fig.subplots_adjust(left=0.004, right=0.996, top=0.996, bottom=0.004)
fig.savefig('architecture_best_run.png', dpi=300, facecolor='white')
fig.savefig('architecture_best_run.pdf', facecolor='white')
fig.savefig('architecture_best_run.svg', facecolor='white')
print('saved: architecture_best_run.png / .pdf / .svg')
