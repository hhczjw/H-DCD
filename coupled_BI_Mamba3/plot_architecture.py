"""
Coupled-BI-Mamba3 Network Architecture Diagram Generator
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

fig, ax = plt.subplots(figsize=(22, 30))
ax.set_xlim(0, 22)
ax.set_ylim(0, 30)
ax.axis('off')

# ─── Color palette ───
C_TEXT   = '#4A90D9'   # blue
C_AUDIO  = '#E85D75'   # red/pink
C_VIDEO  = '#50B86C'   # green
C_FUSE   = '#9B6FB0'   # purple
C_GATE   = '#F5A623'   # orange
C_SSM    = '#2EC4B6'   # teal
C_HEAD   = '#E0E0E0'   # light gray
C_ARROW  = '#555555'
C_BG     = '#FAFAFA'

def draw_box(x, y, w, h, color, text, fontsize=8, textcolor='white', alpha=0.9, bold=False):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                         facecolor=color, edgecolor='#333', linewidth=1.2, alpha=alpha)
    ax.add_patch(box)
    weight = 'bold' if bold else 'normal'
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fontsize, color=textcolor, weight=weight, family='sans-serif')
    return (x + w/2, y + h/2)

def draw_arrow(x1, y1, x2, y2, color=C_ARROW, lw=1.5, style='->', connectionstyle='arc3,rad=0'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                connectionstyle=connectionstyle))

def draw_text(x, y, text, fontsize=7, color='#333', ha='center', va='center', style='normal'):
    ax.text(x, y, text, ha=ha, va=va, fontsize=fontsize, color=color,
            family='monospace' if style == 'mono' else 'sans-serif',
            style='italic' if style == 'italic' else 'normal')

# ─── Title ───
ax.text(11, 29.5, 'Coupled-BI-Mamba3: Network Architecture',
        ha='center', va='center', fontsize=16, fontweight='bold', color='#222')

# ═══════════════════════════════════════
# LAYER 1: Input Encoders (y ≈ 27-28)
# ═══════════════════════════════════════
draw_text(11, 28.3, '① Input Encoding & Projection', fontsize=11, color='#444', style='italic')

# Text
draw_box(1.5, 27.0, 4, 0.9, C_TEXT, 'Text: RoBERTa-large\n(B, 3, 50) → (B, 50, 768)', fontsize=7)
draw_box(1.5, 25.8, 4, 0.8, C_TEXT, 'Linear 768→128\n(B, 50, 128)', fontsize=7, alpha=0.75)

# Audio
draw_box(7.5, 27.0, 4, 0.9, C_AUDIO, 'Audio: Data2Vec\n(B, T, 768)', fontsize=7)
draw_box(7.5, 25.8, 4, 0.8, C_AUDIO, 'Linear 768→128\n(B, 50, 128)', fontsize=7, alpha=0.75)

# Video
draw_box(13.5, 27.0, 4, 0.9, C_VIDEO, 'Video: Raw Features\n(B, 50, 20)', fontsize=7)
draw_box(13.5, 25.8, 4, 0.8, C_VIDEO, 'Linear 20→128\n(B, 50, 128)', fontsize=7, alpha=0.75)

# Arrows: encoder -> projection
draw_arrow(3.5, 27.0, 3.5, 26.6)
draw_arrow(9.5, 27.0, 9.5, 26.6)
draw_arrow(15.5, 27.0, 15.5, 26.6)

# ═══════════════════════════════════════
# LAYER 2: Sequence Alignment (y ≈ 25.0)
# ═══════════════════════════════════════
draw_text(11, 25.3, '② Sequence Alignment (adaptive_avg_pool1d → L=50)', fontsize=8, color='#666')

# Arrows down from projection to ISM
draw_arrow(3.5, 25.8, 3.5, 24.6)
draw_arrow(9.5, 25.8, 9.5, 24.6)
draw_arrow(15.5, 25.8, 15.5, 24.6)

# ═══════════════════════════════════════
# LAYER 3: ISM (y ≈ 21-24.5)
# ═══════════════════════════════════════
draw_text(11, 24.4, '③ ISM: Intra-modal Sequence Modeling (× ism_depth)', fontsize=11, color='#444', style='italic')

ism_y = 20.5
ism_h = 3.5

# ISM Text
draw_box(1.0, ism_y, 5, ism_h, '#3A7BBF', '', alpha=0.15)
draw_text(3.5, ism_y + ism_h - 0.25, 'ISM (Text)', fontsize=8, color=C_TEXT, style='italic')
# ISM internal blocks
for i in range(3):
    by = ism_y + 0.3 + i * 1.05
    draw_box(1.3, by, 4.4, 0.85, '#5B9BD5',
             f'Block_GLCE [{i}]\nGLCE → BiMamba → BSSM Gate', fontsize=6, alpha=0.85)

# ISM Audio
draw_box(7.0, ism_y, 5, ism_h, '#C0485A', '', alpha=0.15)
draw_text(9.5, ism_y + ism_h - 0.25, 'ISM (Audio)', fontsize=8, color=C_AUDIO, style='italic')
for i in range(3):
    by = ism_y + 0.3 + i * 1.05
    draw_box(7.3, by, 4.4, 0.85, '#E8899A',
             f'Block_GLCE [{i}]\nGLCE → BiMamba → BSSM Gate', fontsize=6, alpha=0.85)

# ISM Video
draw_box(13.0, ism_y, 5, ism_h, '#3D9B58', '', alpha=0.15)
draw_text(15.5, ism_y + ism_h - 0.25, 'ISM (Video)', fontsize=8, color=C_VIDEO, style='italic')
for i in range(3):
    by = ism_y + 0.3 + i * 1.05
    draw_box(13.3, by, 4.4, 0.85, '#7BC98E',
             f'Block_GLCE [{i}]\nGLCE → BiMamba → BSSM Gate', fontsize=6, alpha=0.85)

# Arrows from alignment to ISM
draw_arrow(3.5, 24.6, 3.5, ism_y + ism_h)
draw_arrow(9.5, 24.6, 9.5, ism_y + ism_h)
draw_arrow(15.5, 24.6, 15.5, ism_y + ism_h)

# ISM detail annotation
draw_text(19.5, ism_y + 1.5,
          'Block_GLCE:\n'
          '┌ RMSNorm\n'
          '├ GLCE (Global+Local)\n'
          '├ RMSNorm\n'
          '├ BiMamba (fwd+bwd)\n'
          '└ BSSM Gate\n'
          '    SiLU(Linear(x))\n'
          '    ⊙ x',
          fontsize=5.5, color='#555', ha='left', va='center', style='mono')

# ═══════════════════════════════════════
# LAYER 4: CoupledMamba3Fork (y ≈ 15-20)
# ═══════════════════════════════════════
draw_arrow(3.5, ism_y, 3.5, 19.5)
draw_arrow(9.5, ism_y, 9.5, 19.5)
draw_arrow(15.5, ism_y, 15.5, 19.5)

draw_text(11, 19.5, '④ CoupledMamba3Fork (× num_layers)', fontsize=11, color='#444', style='italic')

cm_y = 15.0
cm_h = 4.0

# Background box for Coupled Mamba
draw_box(0.5, cm_y, 18, cm_h, '#8B5FBF', '', alpha=0.08)

# --- Layer 0 ---
layer0_y = cm_y + 2.2
draw_text(1.2, layer0_y + 0.65, 'Layer 0', fontsize=7, color='#666')

# Three CrossMamba3Cells
draw_box(2.5, layer0_y, 4.5, 0.9, '#7B5EA7',
         'CrossMamba3Cell (Audio)\nQ←Audio, K/V←Text⊕Video', fontsize=6)
draw_box(7.5, layer0_y, 4.5, 0.9, '#7B5EA7',
         'CrossMamba3Cell (Video)\nQ←Video, K/V←Text⊕Audio', fontsize=6)
draw_box(12.5, layer0_y, 4.5, 0.9, '#7B5EA7',
         'CrossMamba3Cell (Text)\nQ←Text, K/V←Audio⊕Video', fontsize=6)

# Dynamic Source Weighting
draw_text(11, layer0_y - 0.15, 'Dynamic Source Weighting: w = softmax(MLP(cat(tgt, src0, src1)))',
          fontsize=6, color='#777')

# --- Layer 1 ---
layer1_y = cm_y + 0.5
draw_text(1.2, layer1_y + 0.65, 'Layer 1', fontsize=7, color='#666')

draw_box(2.5, layer1_y, 4.5, 0.9, '#6A4F96',
         'CrossMamba3Cell (Audio)\nQ←Audio, K/V←Text⊕Video', fontsize=6)
draw_box(7.5, layer1_y, 4.5, 0.9, '#6A4F96',
         'CrossMamba3Cell (Video)\nQ←Video, K/V←Text⊕Audio', fontsize=6)
draw_box(12.5, layer1_y, 4.5, 0.9, '#6A4F96',
         'CrossMamba3Cell (Text)\nQ←Text, K/V←Audio⊕Video', fontsize=6)

# Arrows between layers
draw_arrow(4.75, layer0_y, 4.75, layer1_y + 0.9, color='#9B6FB0', lw=1)
draw_arrow(9.75, layer0_y, 9.75, layer1_y + 0.9, color='#9B6FB0', lw=1)
draw_arrow(14.75, layer0_y, 14.75, layer1_y + 0.9, color='#9B6FB0', lw=1)

# CrossMamba3Cell detail annotation
draw_text(19.5, cm_y + 2.0,
          'CrossMamba3Cell:\n'
          '┌ tgt → Q(C), dt, A\n'
          '├ src → K(B), V(x)\n'
          '├ w = softmax(MLP)\n'
          '├ K = w₀K₀ + w₁K₁\n'
          '├ V = w₀V₀ + w₁V₁\n'
          '├ Mamba-3 SSM\n'
          '│  (Q,K,V,ADT,DT,\n'
          '│   Trap,RoPE,D)\n'
          '└ out_proj(y)',
          fontsize=5.5, color='#555', ha='left', va='center', style='mono')

# ═══════════════════════════════════════
# LAYER 5: GCMN Gate (y ≈ 12.5-14.5)
# ═══════════════════════════════════════
draw_arrow(3.5, cm_y, 3.5, 14.0)
draw_arrow(9.5, cm_y, 9.5, 14.0)
draw_arrow(15.5, cm_y, 15.5, 14.0)

draw_text(11, 14.0, '⑤ GCMN Gate (Gated Cross-Modal Fusion)', fontsize=11, color='#444', style='italic')

gcmn_y = 11.5
gcmn_h = 2.2

draw_box(0.5, gcmn_y, 18, gcmn_h, C_GATE, '', alpha=0.1)

# Unimodal BSSM
draw_box(1.0, gcmn_y + 1.0, 5, 0.85, '#F0A030',
         'UnimodalBSSM (×3)\nLN → Linear → SiLU → Linear + Res', fontsize=6, alpha=0.85)

# CrossModal BSSM
draw_box(7.0, gcmn_y + 1.0, 5, 0.85, '#E09020',
         'CrossModalBSSM\nconcat(L,A,V) → MLP → decouple', fontsize=6, alpha=0.85)

# Gated Fusion
draw_box(13.0, gcmn_y + 1.0, 5, 0.85, '#D08010',
         'Gated Fusion\ngate = σ(MLP)\nout = uni + gate ⊙ cross', fontsize=6, alpha=0.85)

# Flow arrows inside GCMN
draw_arrow(3.5, gcmn_y + 1.0, 3.5, gcmn_y + 0.7)
draw_arrow(9.5, gcmn_y + 1.0, 9.5, gcmn_y + 0.7)
draw_arrow(15.5, gcmn_y + 1.0, 15.5, gcmn_y + 0.7)

draw_text(11, gcmn_y + 0.35, 'LayerNorm → output (per modality)',
          fontsize=7, color='#888')

# GCMN detail
draw_text(19.5, gcmn_y + 1.1,
          'GCMN:\n'
          '┌ Unimodal path\n'
          '│  (preserve modality)\n'
          '├ CrossModal path\n'
          '│  (3d → 6d → 3d)\n'
          '└ Gate: σ(MLP)\n'
          '    out = uni + g⊙cross',
          fontsize=5.5, color='#555', ha='left', va='center', style='mono')

# ═══════════════════════════════════════
# LAYER 6: Dialogue Context (y ≈ 8.5-11)
# ═══════════════════════════════════════
draw_arrow(3.5, gcmn_y, 3.5, 10.5)
draw_arrow(9.5, gcmn_y, 9.5, 10.5)
draw_arrow(15.5, gcmn_y, 15.5, 10.5)

draw_text(11, 10.5, '⑥ Dialogue Context Fusion (Bidirectional)', fontsize=11, color='#444', style='italic')

ctx_y = 7.8
ctx_h = 2.4

draw_box(0.5, ctx_y, 18, ctx_h, '#607D8B', '', alpha=0.08)

# Main utterance
draw_box(1.5, ctx_y + 1.2, 7, 0.85, '#78909C',
         'Main Utterance\nmean_pool → (B, 128) per modality', fontsize=6, alpha=0.85)

# Context utterance
draw_box(9.5, ctx_y + 1.2, 7, 0.85, '#90A4AE',
         'Context Utterance\nmean_pool → (B, 128) per modality', fontsize=6, alpha=0.85)

# Fusion
draw_box(3.5, ctx_y + 0.1, 12, 0.85, '#546E7A',
         'stack([ctx, main]) → CoupledMamba3Fork → t=1\n'
         'Forward: ctx→main  |  Reverse: main→ctx  |  Concat → (B, 768)',
         fontsize=6, alpha=0.85)

draw_arrow(5.0, ctx_y + 1.2, 5.0, ctx_y + 0.95)
draw_arrow(13.0, ctx_y + 1.2, 13.0, ctx_y + 0.95)

# ═══════════════════════════════════════
# LAYER 7: Output Head (y ≈ 5.5-7.5)
# ═══════════════════════════════════════
draw_arrow(3.5, ctx_y, 7.5, 6.8)
draw_arrow(9.5, ctx_y, 9.5, 6.8)
draw_arrow(15.5, ctx_y, 11.5, 6.8)

draw_text(11, 7.3, '⑦ Output Head', fontsize=11, color='#444', style='italic')

draw_box(3.0, 5.5, 13, 1.5, '#37474F', '', alpha=0.9)
draw_text(9.5, 6.5, 'Mean Pooling → Concat [T, A, V] → (B, 768)',
          fontsize=8, color='white')
draw_text(9.5, 5.9, 'LayerNorm → Linear(768→128) → GELU → Dropout → Linear(128→1)',
          fontsize=8, color='white')

# Final output
draw_box(7.5, 4.0, 4, 1.0, '#263238',
         'Sentiment Score\nŷ ∈ ℝ (regression)', fontsize=8, textcolor='white', bold=True)
draw_arrow(9.5, 5.5, 9.5, 5.0)

# ═══════════════════════════════════════
# Sub-loss supervision (side annotation)
# ═══════════════════════════════════════
draw_text(19.5, ism_y - 0.5,
          'Sub-loss Heads:\n'
          '┌ fc_T: Linear(128→1)\n'
          '├ fc_A: Linear(128→1)\n'
          '└ fc_V: Linear(128→1)\n'
          '(deep supervision\n'
          ' on ISM CLS tokens)',
          fontsize=5.5, color='#555', ha='left', va='center', style='mono')

# Sub-loss arrows
draw_arrow(3.5, ism_y, 0.3, ism_y - 0.3, color=C_TEXT, lw=1, style='->', connectionstyle='arc3,rad=0.3')
draw_arrow(9.5, ism_y, 0.3, ism_y - 0.5, color=C_AUDIO, lw=1, style='->', connectionstyle='arc3,rad=0.3')
draw_arrow(15.5, ism_y, 0.3, ism_y - 0.7, color=C_VIDEO, lw=1, style='->', connectionstyle='arc3,rad=0.3')

# ═══════════════════════════════════════
# Legend
# ═══════════════════════════════════════
legend_y = 3.2
legend_items = [
    (C_TEXT, 'Text Modality'),
    (C_AUDIO, 'Audio Modality'),
    (C_VIDEO, 'Video Modality'),
    (C_FUSE, 'Cross-Modal Fusion'),
    (C_GATE, 'Gated Fusion (GCMN)'),
    (C_SSM, 'SSM / Mamba'),
]
for i, (color, label) in enumerate(legend_items):
    x = 2 + i * 3.3
    draw_box(x, legend_y, 0.4, 0.3, color, '', alpha=0.9)
    draw_text(x + 0.6, legend_y + 0.15, label, fontsize=6.5, color='#333', ha='left')

# Loss function annotation
draw_text(11, 2.6, 'Loss = SmoothL1(ŷ,y) + α·DiscreteCE(7-cls) + β·Σ SmoothL1(sub_ŷ, y)',
          fontsize=7, color='#888', style='italic')

# Dimension annotations on the right side
draw_text(19.5, 27.4, 'Feature Dims:\nText: 768\nAudio: 768\nVideo: 20',
          fontsize=6, color='#666', ha='left', style='mono')
draw_text(19.5, 25.8, 'After Proj:\nd_model = 128',
          fontsize=6, color='#666', ha='left', style='mono')
draw_text(19.5, 19.5, 'd_state = 64\nexpand = 2\nheaddim = 32\nnheads = 8',
          fontsize=6, color='#666', ha='left', style='mono')

plt.tight_layout()
plt.savefig('/home/zjw/桌面/H-DCD/coupled_BI_Mamba3/architecture_diagram.png',
            dpi=200, bbox_inches='tight', facecolor='white')
plt.savefig('/home/zjw/桌面/H-DCD/coupled_BI_Mamba3/architecture_diagram.pdf',
            bbox_inches='tight', facecolor='white')
print("Saved: architecture_diagram.png / .pdf")
