# -*- coding: utf-8 -*-
"""
merlin_data.py – data for the robust model: the same formulas rendered in many fonts, plus image augmentation.

Used only for training (Colab notebook). The web app does not need this file.

1) render_latex_batch  – renders training formulas with pdflatex in other LaTeX fonts
                          (Times, Palatino, Helvetica-math, Euler, Fourier, …) and in a bold variant
2) render_mathtext      – the same idea through matplotlib (DejaVu Sans/Serif, STIX, STIX Sans)
3) Augmenter            – random margins, stroke width, blur, rotation, JPEG, noise, contrast
"""
import io
import math
import os
import random
import re
import shutil
import subprocess
import tempfile

import numpy as np
from PIL import Image, ImageFilter, ImageOps

# ------------------------------------------------------------------------------------------
#  LaTeX fonts: name -> (preamble, bold)
# ------------------------------------------------------------------------------------------
FONT_PRESETS = {
    'cm':           ('', False),                                                   # Computer Modern
    'cm_bold':      ('', True),
    'times':        ('\\usepackage{mathptmx}', False),                             # Times
    'times_bold':   ('\\usepackage{mathptmx}', True),
    'palatino':     ('\\usepackage{mathpazo}', False),                             # Palatino
    'helvetica':    ('\\usepackage[helvet]{sfmath}\\renewcommand{\\familydefault}{\\sfdefault}', False),
    'helvetica_bold': ('\\usepackage[helvet]{sfmath}\\renewcommand{\\familydefault}{\\sfdefault}', True),
    'lmodern':      ('\\usepackage{lmodern}', False),                              # Latin Modern
    'euler':        ('\\usepackage{eulervm}', False),                              # Euler (handwriting-like)
    'fourier':      ('\\usepackage{fourier}', False),                              # Utopia
    'cmbright':     ('\\usepackage{cmbright}', False),                             # sans serif
    'kpfonts':      ('\\usepackage{kpfonts}', False),
    'newtx':        ('\\usepackage{newtxtext,newtxmath}', False),                  # Times (modern)
    'newpx':        ('\\usepackage{newpxtext,newpxmath}', False),                  # Palatino (modern)
    'arev':         ('\\usepackage{arev}', False),                                 # Bitstream Vera, sans serif
    'concrete':     ('\\usepackage{concmath}', False),
}
MATHTEXT_FONTS = ['dejavusans', 'dejavuserif', 'stix', 'stixsans']

DOC_HEAD = r"""\documentclass[12pt]{article}
\usepackage{amsmath,amssymb}
%s
\usepackage[active,tightpage]{preview}
\setlength\PreviewBorder{2pt}
\pagestyle{empty}
\begin{document}
"""


def _run(cmd, cwd, timeout):
    return subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)


def available_fonts(candidates=None):
    """Returns the fonts whose packages exist and compile a test formula (depends on the installed TeX)."""
    ok = []
    for name in (candidates or FONT_PRESETS):
        pre, bold = FONT_PRESETS[name]
        with tempfile.TemporaryDirectory() as d:
            body = ('\\begin{preview}' + ('\\boldmath' if bold else '') +
                    '$\\displaystyle \\frac{\\alpha^2}{\\sqrt{x_1}} + \\sum_{i=1}^n \\mathcal{L}$\\end{preview}\n')
            open(os.path.join(d, 'f.tex'), 'w').write(DOC_HEAD % pre + body + '\\end{document}\n')
            try:
                r = _run(['pdflatex', '-interaction=nonstopmode', '-halt-on-error', 'f.tex'], d, 60)
                if r.returncode == 0 and os.path.exists(os.path.join(d, 'f.pdf')):
                    ok.append(name)
            except Exception:
                pass
    return ok


def crop_to_ink(img, thr=200, pad=2):
    g = np.asarray(img.convert('L'))
    ink = g < thr
    if not ink.any():
        return None
    r, c = np.where(ink.any(1))[0], np.where(ink.any(0))[0]
    g = g[max(0, r[0] - pad):r[-1] + 1 + pad, max(0, c[0] - pad):c[-1] + 1 + pad]
    return Image.fromarray(g)


def render_latex_batch(formulas, font, dpi=150, timeout=300):
    """Renders a list of formulas into one PDF (one formula = one page) and returns a list of
    PIL images (None for formulas that failed). Formulas with a LaTeX error are detected from the log
    (each formula is on its own line) and dropped, so a label and its image never get out of sync."""
    pre, bold = FONT_PRESETS[font]
    head = DOC_HEAD % pre
    first_line = head.count('\n') + 1
    lines = ['\\begin{preview}' + ('\\boldmath' if bold else '') + '$\\displaystyle ' + f + '$\\end{preview}'
             for f in formulas]
    out = [None] * len(formulas)
    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, 'f.tex'), 'w', encoding='utf-8').write(head + '\n'.join(lines) + '\n\\end{document}\n')
        try:
            _run(['pdflatex', '-interaction=batchmode', 'f.tex'], d, timeout)
        except subprocess.TimeoutExpired:
            return out
        pdf, log = os.path.join(d, 'f.pdf'), os.path.join(d, 'f.log')
        if not os.path.exists(pdf):
            return out
        bad = set()
        txt = open(log, encoding='latin-1').read() if os.path.exists(log) else ''
        for m in re.finditer(r'^! .*?^l\.(\d+)', txt, flags=re.M | re.S):
            idx = int(m.group(1)) - first_line
            for j in (idx - 1, idx):                    # to be safe: also the previous formula (unclosed brace)
                if 0 <= j < len(formulas):
                    bad.add(j)
        info = _run(['pdfinfo', pdf], d, 60).stdout.decode('latin-1')
        pages = int(re.search(r'Pages:\s+(\d+)', info).group(1)) if 'Pages:' in info else 0
        if pages != len(formulas):                       # some formula produced no page -> do not take the risk
            if len(formulas) == 1:
                return out
            h = len(formulas) // 2                       # split in half until the problem is isolated
            return render_latex_batch(formulas[:h], font, dpi, timeout) + \
                render_latex_batch(formulas[h:], font, dpi, timeout)
        _run(['pdftoppm', '-r', str(dpi), '-gray', '-png', 'f.pdf', 'p'], d, timeout)
        files = sorted((f for f in os.listdir(d) if f.startswith('p-') and f.endswith('.png')),
                       key=lambda s: int(re.search(r'(\d+)', s).group(1)))
        for i, fn in enumerate(files[:len(formulas)]):
            if i in bad:
                continue
            with Image.open(os.path.join(d, fn)) as im:
                out[i] = crop_to_ink(im.convert('L'))
    return out


def render_mathtext(formula, fontset='dejavusans', dpi=150):
    """Rendering through matplotlib mathtext (no LaTeX). Formulas with commands mathtext does not know
    raise an error -> None."""
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    # mathtext misreads spaces around ^, _ and { (e.g. "q ^ { 2 }"), so the source is compacted first
    f = re.sub(r'\s*([\^_])\s*', r'\1', formula)
    f = re.sub(r'\s*([{}])\s*', r'\1', f)
    try:
        fig = Figure(figsize=(0.01, 0.01))
        FigureCanvasAgg(fig)
        fig.text(0, 0, '$' + f + '$', fontsize=14, math_fontfamily=fontset)
        buf = io.BytesIO()
        fig.savefig(buf, dpi=dpi, format='png', bbox_inches='tight', pad_inches=0.05, facecolor='white')
        buf.seek(0)
        return crop_to_ink(Image.open(buf).convert('L'))
    except Exception:
        return None


# ------------------------------------------------------------------------------------------
#  Augmentation
# ------------------------------------------------------------------------------------------
class Augmenter:
    """Random changes that resemble real screenshots: size and margins, stroke width,
    blur, slight rotation, low resolution, JPEG, noise and contrast. Works on a greyscale PIL image."""
    def __init__(self, p=0.7, seed=None, margin_always=False, rng=None):
        # rng=random (the global module) in a DataLoader: PyTorch seeds it separately in every worker
        self.p, self.margin_always = p, margin_always
        self.rng = rng if rng is not None else random.Random(seed)

    def margins(self, img):
        r = self.rng
        w, h = img.size
        frac = r.uniform(0.3, 0.95)                      # share of the height taken by the formula
        v = int(round(h * (1 / frac - 1) / 2))
        left, right = r.randint(2, max(3, h // 3)), r.randint(2, max(3, h))
        top = int(round(v * r.uniform(0.6, 1.4))); bottom = max(0, 2 * v - top)
        return ImageOps.expand(img, (left, top, right, bottom), fill=255)

    def __call__(self, img):
        r = self.rng
        img = img.convert('L')
        if self.margin_always or r.random() < 0.5:
            img = self.margins(img)
        if r.random() > self.p:
            return img
        base = img
        if r.random() < 0.3 and min(img.size) > 12:
            img = img.filter(ImageFilter.MinFilter(3))      # thicker strokes (dark ink spreads)
        if r.random() < 0.3:
            img = img.rotate(r.uniform(-2.0, 2.0), resample=Image.BILINEAR, expand=True, fillcolor=255)
        if r.random() < 0.3:
            s = r.uniform(0.35, 0.8)                        # down to low resolution and back
            w, h = img.size
            img = img.resize((max(8, int(w * s)), max(8, int(h * s))), Image.BILINEAR).resize((w, h), Image.BILINEAR)
        if r.random() < 0.3:
            img = img.filter(ImageFilter.GaussianBlur(r.uniform(0.3, 1.1)))
        if r.random() < 0.3:                                # grey background / faint ink
            a = np.asarray(img, dtype=np.float32) / 255.0
            bg, fg = r.uniform(190, 255), r.uniform(0, 90)
            img = Image.fromarray((fg + a * (bg - fg)).clip(0, 255).astype(np.uint8))
        if r.random() < 0.2:
            a = np.asarray(img, dtype=np.float32)
            a = a + np.random.default_rng(r.randrange(1 << 30)).normal(0, r.uniform(3, 12), a.shape)
            img = Image.fromarray(a.clip(0, 255).astype(np.uint8))
        if r.random() < 0.3:
            buf = io.BytesIO()
            img.save(buf, format='JPEG', quality=r.randint(25, 85))
            buf.seek(0)
            img = Image.open(buf).convert('L')
        # safeguard: if the formula has almost vanished (too blurred/faded), return the undistorted image
        ink0 = (np.asarray(base) < 128).sum()
        if ink0 > 0 and (np.asarray(img) < 128).sum() < 0.3 * ink0:
            return base
        return img


# ------------------------------------------------------------------------------------------
#  Workers for parallel rendering (ProcessPoolExecutor)
# ------------------------------------------------------------------------------------------
def latex_job(args):
    """(font, dpi, [(id, formula)], out_dir, prefix) -> list of ids that were saved."""
    font, dpi, items, out_dir, prefix = args
    ims = render_latex_batch([f for _, f in items], font, dpi=dpi)
    done = []
    for (i, _), im in zip(items, ims):
        if im is not None and im.width > 4 and im.height > 4:
            im.save(os.path.join(out_dir, f'{prefix}{i}.png'))
            done.append(i)
    return done


def mathtext_job(args):
    """(fontset, dpi, [(id, formula)], out_dir, prefix) -> list of ids that were saved."""
    fontset, dpi, items, out_dir, prefix = args
    done = []
    for i, f in items:
        im = render_mathtext(f, fontset, dpi=dpi)
        if im is not None and im.width > 4 and im.height > 4:
            im.save(os.path.join(out_dir, f'{prefix}{i}.png'))
            done.append(i)
    return done
