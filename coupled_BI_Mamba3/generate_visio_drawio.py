"""
Generate Coupled-BI-Mamba3 architecture diagram in .drawio format.
Open with draw.io or import into Visio.
"""
import xml.etree.ElementTree as ET
from xml.dom import minidom

# ── Colors ──
COL_TEXT  = '#4A90D9'
COL_AUDIO = '#E85D75'
COL_VIDEO = '#50B86C'
COL_ISM_T = '#5B9BD5'
COL_ISM_A = '#E8899A'
COL_ISM_V = '#7BC98E'
COL_FUSE  = '#7B5EA7'
COL_GCMN  = '#F5A623'
COL_HEAD  = '#37474F'
COL_BG    = '#FAFAFA'
COL_LIGHT = '#F0F0F0'

mxfile = ET.Element('mxfile', host='app.diagrams.net', type='device')
diagram = ET.SubElement(mxfile, 'diagram', name='CoupledBI-Mamba3', id='arch')
model = ET.SubElement(diagram, 'mxGraphModel', dx='1422', dy='762', grid='1', gridSize='10',
                       guides='1', tooltips='1', connect='1', arrows='1', fold='1',
                       page='1', pageScale='1', pageWidth='1600', pageHeight='2400',
                       math='0', shadow='0')
root = ET.SubElement(model, 'root')
ET.SubElement(root, 'mxCell', id='0')
ET.SubElement(root, 'mxCell', id='1', parent='0')

_id = [2]  # mutable counter
def nid():
    _id[0] += 1
    return str(_id[0])

def box(x, y, w, h, label, fill='#4A90D9', text_color='#ffffff', fontsize='12',
        stroke='#333333', style='rounded=1;whiteSpace=wrap;html=1;'):
    cell_id = nid()
    styles = f'fillColor={fill};strokeColor={stroke};fontColor={text_color};fontSize={fontsize};' + style
    c = ET.SubElement(root, 'mxCell', id=cell_id, value=label, style=styles,
                      vertex='1', parent='1')
    ET.SubElement(c, 'mxGeometry', x=str(x), y=str(y), width=str(w), height=str(h)).set('as', 'geometry')
    return cell_id

def arrow(src, tgt, style='rounded=1;orthogonalLoop=1;jettySize=auto;html=1;'):
    cell_id = nid()
    c = ET.SubElement(root, 'mxCell', id=cell_id, value='', style=f'strokeColor=#555555;strokeWidth=1.5;{style}',
                      edge='1', parent='1', source=src, target=tgt)
    ET.SubElement(c, 'mxGeometry', relative='1').set('as', 'geometry')

def label(x, y, w, h, text, fontsize='11', color='#444444', align='center', style=''):
    cell_id = nid()
    s = f'html=1;align={align};verticalAlign=middle;fontSize={fontsize};fontColor={color};{style}'
    c = ET.SubElement(root, 'mxCell', id=cell_id, value=text, style=s, vertex='1', parent='1')
    ET.SubElement(c, 'mxGeometry', x=str(x), y=str(y), width=str(w), height=str(h)).set('as', 'geometry')
    return cell_id

# ══════════════════════════════════════════════════
# Title
# ══════════════════════════════════════════════════
label(400, 20, 800, 40, 'Coupled-BI-Mamba3: Network Architecture', fontsize='18', color='#222222',
      style='fontStyle=1;')

# ══════════════════════════════════════════════════
# ① Input Encoding & Projection
# ══════════════════════════════════════════════════
label(400, 80, 800, 30, '① Input Encoding & Projection', fontsize='14', color='#666')

# Text encoders
enc_t = box(80, 130, 260, 50, 'Text: RoBERTa-large<br>(B, 3, 50) → (B, 50, 768)',
            fill=COL_TEXT, fontsize='11')
proj_t = box(80, 200, 260, 45, 'Linear 768→128<br>(B, 50, 128)',
             fill='#6AADE0', fontsize='11')

# Audio encoders
enc_a = box(430, 130, 260, 50, 'Audio: Data2Vec<br>(B, T, 768)',
            fill=COL_AUDIO, fontsize='11')
proj_a = box(430, 200, 260, 45, 'Linear 768→128<br>(B, 50, 128)',
             fill='#F09BAE', fontsize='11')

# Video encoders
enc_v = box(780, 130, 260, 50, 'Video: Raw Features<br>(B, 50, 20)',
            fill=COL_VIDEO, fontsize='11')
proj_v = box(780, 200, 260, 45, 'Linear 20→128<br>(B, 50, 128)',
             fill='#7DCB97', fontsize='11')

arrow(enc_t, proj_t)
arrow(enc_a, proj_a)
arrow(enc_v, proj_v)

# ══════════════════════════════════════════════════
# ② Sequence Alignment
# ══════════════════════════════════════════════════
align_bg = box(280, 265, 560, 35,
               '② Sequence Alignment  (adaptive_avg_pool1d → L=50)',
               fill=COL_LIGHT, text_color='#555', stroke='#cccccc', fontsize='11')

arrow(proj_t, align_bg)
arrow(proj_a, align_bg)
arrow(proj_v, align_bg)

# ══════════════════════════════════════════════════
# ③ ISM (Intra-modal Sequence Modeling)
# ══════════════════════════════════════════════════
label(680, 320, 800, 30, '③ ISM: Intra-modal Sequence Modeling (× ism_depth)', fontsize='14', color='#666')

ism_h = 130
ism_w = 250
ism_y = 360

# ISM Text
ism_t_bg = box(80, ism_y, ism_w, ism_h, '', fill=COL_ISM_T, stroke=COL_TEXT, style='rounded=1;whiteSpace=wrap;html=1;opacity=30;')
label(80, ism_y, ism_w, 25, 'ISM (Text)', fontsize='10', color=COL_TEXT, style='fontStyle=1;')
b1 = box(95, ism_y + 30, ism_w - 30, 28, 'Block_GLCE [0]: RMSNorm → GLCE → BiMamba → BSSM Gate',
         fill=COL_ISM_T, fontsize='8')
b2 = box(95, ism_y + 62, ism_w - 30, 28, 'Block_GLCE [1]: RMSNorm → GLCE → BiMamba → BSSM Gate',
         fill=COL_ISM_T, fontsize='8')
b3 = box(95, ism_y + 94, ism_w - 30, 28, 'Block_GLCE [2]: RMSNorm → GLCE → BiMamba → BSSM Gate',
         fill=COL_ISM_T, fontsize='8')

# ISM Audio
ism_a_bg = box(420, ism_y, ism_w, ism_h, '', fill=COL_ISM_A, stroke=COL_AUDIO, style='rounded=1;whiteSpace=wrap;html=1;opacity=30;')
label(420, ism_y, ism_w, 25, 'ISM (Audio)', fontsize='10', color=COL_AUDIO, style='fontStyle=1;')
b4 = box(435, ism_y + 30, ism_w - 30, 28, 'Block_GLCE [0]: RMSNorm → GLCE → BiMamba → BSSM Gate',
         fill=COL_ISM_A, fontsize='8')
b5 = box(435, ism_y + 62, ism_w - 30, 28, 'Block_GLCE [1]: RMSNorm → GLCE → BiMamba → BSSM Gate',
         fill=COL_ISM_A, fontsize='8')
b6 = box(435, ism_y + 94, ism_w - 30, 28, 'Block_GLCE [2]: RMSNorm → GLCE → BiMamba → BSSM Gate',
         fill=COL_ISM_A, fontsize='8')

# ISM Video
ism_v_bg = box(760, ism_y, ism_w, ism_h, '', fill=COL_ISM_V, stroke=COL_VIDEO, style='rounded=1;whiteSpace=wrap;html=1;opacity=30;')
label(760, ism_y, ism_w, 25, 'ISM (Video)', fontsize='10', color=COL_VIDEO, style='fontStyle=1;')
b7 = box(775, ism_y + 30, ism_w - 30, 28, 'Block_GLCE [0]: RMSNorm → GLCE → BiMamba → BSSM Gate',
         fill=COL_ISM_V, fontsize='8')
b8 = box(775, ism_y + 62, ism_w - 30, 28, 'Block_GLCE [1]: RMSNorm → GLCE → BiMamba → BSSM Gate',
         fill=COL_ISM_V, fontsize='8')
b9 = box(775, ism_y + 94, ism_w - 30, 28, 'Block_GLCE [2]: RMSNorm → GLCE → BiMamba → BSSM Gate',
         fill=COL_ISM_V, fontsize='8')

# ISM detail annotation (right side)
label(1060, ism_y + 10, 440, 130,
      '<b>Block_GLCE 内部结构:</b><br>'
      '<font color="#666">'
      '1. Fused Add + RMSNorm<br>'
      '2. GLCE (Global: Linear + Local: Conv1d k=3)<br>'
      '3. RMSNorm<br>'
      '4. BiMamba (fwd+bwd 双向SSM扫描)<br>'
      '   out = out_fwd + out_bwd.flip()<br>'
      '5. BSSM Gate: SiLU(Linear(x→2x))→Linear→SiLU → ⊙ x<br>'
      '</font>',
      fontsize='9', color='#444', align='left')

# Sub-loss arrows
sub_loss = label(80, ism_y + ism_h + 5, 250, 25,
                 'Sub-loss: fc_T/fc_A/fc_V: Linear(128→1)', fontsize='8', color='#888')
arrow(b3, sub_loss, style='dashed=1;')
arrow(b6, sub_loss, style='dashed=1;')
arrow(b9, sub_loss, style='dashed=1;')

arrow(align_bg, ism_t_bg)
arrow(align_bg, ism_a_bg)
arrow(align_bg, ism_v_bg)

# ══════════════════════════════════════════════════
# ④ CoupledMamba3Fork
# ══════════════════════════════════════════════════
cm_y = 520
cm_bg = box(30, cm_y, 1020, 200, '', fill='#F3EEFF', stroke=COL_FUSE,
            style='rounded=1;whiteSpace=wrap;html=1;opacity=40;')
label(400, cm_y - 5, 520, 25, '④ CoupledMamba3Fork (× num_layers)', fontsize='14', color='#666')

# Layer 0
label(40, cm_y + 25, 120, 25, 'Layer 0', fontsize='10', color='#777')
cm_l0_t = box(170, cm_y + 25, 270, 40, 'CrossMamba3Cell (Text)<br>Q←Text, K/V←Audio⊕Video',
              fill=COL_FUSE, fontsize='9')
cm_l0_a = box(470, cm_y + 25, 270, 40, 'CrossMamba3Cell (Audio)<br>Q←Audio, K/V←Text⊕Video',
              fill=COL_FUSE, fontsize='9')
cm_l0_v = box(770, cm_y + 25, 270, 40, 'CrossMamba3Cell (Video)<br>Q←Video, K/V←Text⊕Audio',
              fill=COL_FUSE, fontsize='9')

# Layer 1
label(40, cm_y + 80, 120, 25, 'Layer 1', fontsize='10', color='#777')
cm_l1_t = box(170, cm_y + 80, 270, 40, 'CrossMamba3Cell (Text)<br>Q←Text, K/V←Audio⊕Video',
              fill='#6A4F96', fontsize='9')
cm_l1_a = box(470, cm_y + 80, 270, 40, 'CrossMamba3Cell (Audio)<br>Q←Audio, K/V←Text⊕Video',
              fill='#6A4F96', fontsize='9')
cm_l1_v = box(770, cm_y + 80, 270, 40, 'CrossMamba3Cell (Video)<br>Q←Video, K/V←Text⊕Audio',
              fill='#6A4F96', fontsize='9')

# Source Weighting
w_label = label(250, cm_y + 130, 560, 25,
                'Dynamic Source Weighting:  w = softmax(MLP(concat(tgt, src0, src1)))',
                fontsize='10', color='#666', style='fontStyle=2;')

# Mamba-3 Kernel detail
label(30, cm_y + 160, 1020, 35,
      '<b>CrossMamba3Cell 核心:</b> tgt→Q(C), dt, A, trap, angles  |  src→K(B), V(x)  |  '
      'Mamba-3 SSM: ADT = -softplus(A)·softplus(dt)  |  K=B, Q=C, V=x  |  RoPE + Trap + D-skip',
      fontsize='9', color='#555')

arrow(ism_t_bg, cm_l0_t)
arrow(ism_a_bg, cm_l0_a)
arrow(ism_v_bg, cm_l0_v)

arrow(cm_l0_t, cm_l1_t)
arrow(cm_l0_a, cm_l1_a)
arrow(cm_l0_v, cm_l1_v)

# ══════════════════════════════════════════════════
# ⑤ GCMN Gate
# ══════════════════════════════════════════════════
gcmn_y = 740
gcmn_bg = box(30, gcmn_y, 1020, 100, '', fill='#FFF4E0', stroke=COL_GCMN,
              style='rounded=1;whiteSpace=wrap;html=1;opacity=40;')
label(400, gcmn_y - 5, 520, 25, '⑤ GCMN Gate (Gated Cross-Modal Fusion)', fontsize='14', color='#666')

uni = box(50, gcmn_y + 20, 300, 35,
          'UnimodalBSSM (×3):<br>LN → Linear(d→2d) → SiLU → Linear(2d→d) + Res',
          fill='#F0A030', fontsize='9')
cross = box(380, gcmn_y + 20, 300, 35,
            'CrossModalBSSM:<br>concat(L,A,V) → Linear(3d→6d) → SiLU → Linear(6d→3d) → decouple',
            fill='#E09020', fontsize='9')
gate = box(710, gcmn_y + 20, 300, 35,
           'Gated Fusion:<br>gate = σ(MLP(concat(L,A,V)))<br>out = uni + gate ⊙ cross',
           fill='#D08010', fontsize='9')

# Arrows inside GCMN
arrow(cm_l1_t, uni, style='dashed=1;')
arrow(cm_l1_a, uni, style='dashed=1;')
arrow(cm_l1_v, uni, style='dashed=1;')
arrow(cm_l1_t, cross, style='dashed=1;')
arrow(cm_l1_a, cross, style='dashed=1;')
arrow(cm_l1_v, cross, style='dashed=1;')

label(40, gcmn_y + 70, 1020, 25,
      'GCMN公式: o_i = u_i + g_i ⊙ (W · [L;A;V] − u_i),  g_i = σ(h([L;A;V]))',
      fontsize='10', color='#777', style='fontStyle=2;')

# ══════════════════════════════════════════════════
# ⑥ Dialogue Context Fusion
# ══════════════════════════════════════════════════
ctx_y = 860
ctx_bg = box(30, ctx_y, 1020, 100, '', fill='#E8EDF0', stroke='#607D8B',
             style='rounded=1;whiteSpace=wrap;html=1;opacity=40;')
label(400, ctx_y - 5, 520, 25, '⑥ Dialogue Context Fusion (Bidirectional)', fontsize='14', color='#666')

main_u = box(50, ctx_y + 20, 350, 35,
             'Main Utterance<br>mean_pool → (B, 128) per modality',
             fill='#78909C', fontsize='10')
ctx_u = box(440, ctx_y + 20, 350, 35,
            'Context Utterance<br>mean_pool → (B, 128) per modality',
            fill='#90A4AE', fontsize='10')

ctx_fwd = box(100, ctx_y + 65, 350, 30,
              'Forward: stack[ctx → main] → CoupledMamba3Fork → t=1',
              fill='#546E7A', fontsize='9')
ctx_rev = box(500, ctx_y + 65, 350, 30,
              'Reverse: stack[main → ctx] → CoupledMamba3Fork → t=1',
              fill='#546E7A', fontsize='9')
ctx_cat = box(870, ctx_y + 70, 160, 25,
              'Concat → (B, 768)',
              fill='#455A64', text_color='#fff', fontsize='10')

arrow(gcmn_bg, main_u)
arrow(gcmn_bg, ctx_u)
arrow(main_u, ctx_fwd)
arrow(ctx_u, ctx_fwd)
arrow(main_u, ctx_rev)
arrow(ctx_u, ctx_rev)
arrow(ctx_fwd, ctx_cat)
arrow(ctx_rev, ctx_cat)

# ══════════════════════════════════════════════════
# ⑦ Output Head
# ══════════════════════════════════════════════════
head_y = 980
head_bg = box(180, head_y, 720, 55,
              'Mean Pooling → Concat [T, A, V] → (B, 768)<br>'
              'LayerNorm → Linear(768→128) → GELU → Dropout → Linear(128→1)',
              fill=COL_HEAD, fontsize='12')

output = box(360, head_y + 70, 320, 50,
             'Sentiment Score<br>ŷ ∈ ℝ (regression)',
             fill='#263238', fontsize='13', text_color='#ffffff', stroke='#111')

loss = box(200, head_y + 140, 740, 40,
           'Loss = SmoothL1(ŷ,y) + α·DiscreteCE(7-class) + β·Σ SmoothL1(sub_ŷ, y)',
           fill='#F5F5F5', text_color='#666', stroke='#ccc', fontsize='11')

arrow(ctx_cat, head_bg)
arrow(head_bg, output)
arrow(output, loss)

# ══════════════════════════════════════════════════
# Legend
# ══════════════════════════════════════════════════
legend_y = 1170
legend_items = [
    (COL_TEXT,  'Text Modality'),
    (COL_AUDIO, 'Audio Modality'),
    (COL_VIDEO, 'Video Modality'),
    (COL_FUSE,  'Cross-Modal Fusion'),
    (COL_GCMN,  'Gated Fusion (GCMN)'),
]
for i, (c, lbl) in enumerate(legend_items):
    x = 200 + i * 200
    box(x, legend_y, 15, 15, '', fill=c, stroke=c)
    label(x + 20, legend_y, 160, 15, lbl, fontsize='10', color='#333', align='left')

# ══════════════════════════════════════════════════
# Save
# ══════════════════════════════════════════════════
xml_str = minidom.parseString(ET.tostring(mxfile)).toprettyxml(indent='  ')
xml_str = xml_str.replace('<?xml version="1.0" ?>', '<?xml version="1.0" encoding="UTF-8"?>')

out_path = '/home/zjw/桌面/H-DCD/coupled_BI_Mamba3/architecture.drawio'
with open(out_path, 'w', encoding='utf-8') as f:
    f.write(xml_str)

print(f"Saved: {out_path}")
print("Open with draw.io (https://app.diagrams.net) or import into Visio.")
