"""fig5_caption.py — the in-figure caption of Fig. 5 for the review builds (the figures page and the gallery), as the other
main figures carry theirs (their scripts end with draw_justified(fig, CAP_PARAS, fontsize=PS*7.2)).

The legend is read from docs/paper/results_draft.md (from "Figure 5 |" to "Figure 6 |"), so the caption and
the manuscript legend cannot drift apart; it is typeset with pca/figcaption.draw_justified on a canvas exactly as wide as
fig5_model.png and stacked under it. Output: figures/paper_share/Fig5_model.png. The uncaptioned
figures/paper_share/modelling/fig5_model.png is left as is, because the Modelling page prints the legend as text under it
(build_modelling_artifact.py) and the submission build carries no in-figure legend.

Run (after fig5_model.py or any edit of the legend):
    python /home/leon/rnn/paper/fig5_caption.py
"""
import sys
sys.path.insert(0, '/home/leon/rnn/paper')
from style import PS                                   # house rcParams + the Fig. 5 print scale
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
sys.path.insert(0, '/home/leon/dual/pca')
from figcaption import draw_justified

SRC = '/home/leon/dual/figures/paper_share/modelling/fig5_model.png'
OUT = '/home/leon/dual/figures/paper_share/Fig5_model.png'
DRAFT = '/home/leon/dual/docs/paper/results_draft.md'   # the main paper carries the Fig. 5 legend since v12.62
DPI = 400
import os, tempfile
_TMP = os.path.join(tempfile.mkdtemp(prefix='fig5cap_'), 'caption.png')

txt = open(DRAFT, encoding='utf-8').read()
i = txt.index('\nFigure 5 |') + 1
j = txt.index('\nFigure 6 |', i)
paras = [' '.join(p.split()) for p in txt[i:j].strip().split('\n\n') if p.strip()]

im = Image.open(SRC).convert('RGB')
W, H = im.size
HC = 12.0                                              # provisional canvas height (in); cropped to the text below
fig = plt.figure(figsize=(W / DPI, HC), dpi=DPI)
top = 1 - 0.06 / HC                                    # 0.06 in gap under the figure
y_end = draw_justified(fig, paras, fontsize=PS * 7.2, y0=top)
used = (1 - y_end) * HC + 0.10                         # inches of caption, plus a bottom margin
assert used < HC, 'caption longer than the provisional canvas; raise HC'
fig.savefig(_TMP, dpi=DPI, facecolor='white')
plt.close(fig)
cap = Image.open(_TMP).convert('RGB')
assert abs(cap.size[0] - W) <= 2, (cap.size, W)                 # figsize W/DPI can round one pixel short
_c = Image.new('RGB', (W, cap.size[1]), 'white'); _c.paste(cap.crop((0, 0, min(W, cap.size[0]), cap.size[1])), (0, 0)); cap = _c
cap = cap.crop((0, 0, W, int(round(used * DPI))))
outim = Image.new('RGB', (W, H + cap.size[1]), 'white')
outim.paste(im, (0, 0)); outim.paste(cap, (0, H))
outim.save(OUT, dpi=(DPI, DPI), optimize=True)
print(f'{len(paras)} legend paragraphs; caption {cap.size[1]} px; saved {OUT} {outim.size}')
