import re

with open('models/coupled_mamba3_fork.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace file header
content = content.replace('Coupled-Mamba3 Fork', 'Pairwise Cross-Mamba3')
content = content.replace('CrossMamba3Cell', 'PairwiseCrossMamba3Cell')
content = content.replace('CoupledMamba3Fork', 'PairwiseCrossMamba3Fork')

# We need to change the __init__ of PairwiseCrossMamba3Cell
# Remove modality_keys from __init__ because it only deals with one src and one tgt
# b_projs, v_projs should be singular: b_proj_src, v_proj_src
