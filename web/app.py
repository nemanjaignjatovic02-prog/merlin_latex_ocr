# -*- coding: utf-8 -*-
"""MERLIN web app: formula image -> LaTeX. FastAPI backend that serves the model from merlin_model.py."""
import base64
import io
import math
import os
import threading
import time

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from merlin_model import (ImagePreprocessor, beam_search, collate_images, early_exit_generate, feat_stride,
                       load_for_inference, prepare_user_image, score_sequence)

MODEL_DIR = os.environ.get('MODEL_DIR', os.path.join(os.path.dirname(__file__), 'model'))
SAVE_DIR = os.environ.get('SAVE_DIR')       # e.g. SAVE_DIR=_captures -> every prediction is saved to that folder
MAX_BYTES = 8 * 1024 * 1024
MAX_PIXELS = 25_000_000
INK_FRACTIONS = (0.85, 0.6, 0.42, 0.3)        # share of the height taken by the formula (see prepare_user_image)
torch.set_num_threads(max(1, os.cpu_count() or 1))

model = tokenizer = info = preprocessor = None
load_error = None
try:
    model, tokenizer, info = load_for_inference(MODEL_DIR)
    preprocessor = ImagePreprocessor(info['config'])
except Exception as e:                                   # the app starts without a model, but reports it clearly
    load_error = f"{type(e).__name__}: {e}"

lock = threading.Lock()                                  # one request at a time (the Space CPU is small)
app = FastAPI(title="MERLIN – image to LaTeX")
STATIC = os.path.join(os.path.dirname(__file__), 'static')
app.mount('/static', StaticFiles(directory=STATIC), name='static')


@app.get('/')
def index():
    return FileResponse(os.path.join(STATIC, 'index.html'))


@app.get('/api/info')
def api_info():
    if model is None:
        return {'ready': False, 'error': load_error}
    return {'ready': True, 'config': info['config'], 'vocab_size': info['vocab_size'],
            'metrics': info.get('metrics', {}), 'max_gen_len': info.get('max_gen_len', 250)}


def _png_data_url(arr_01):
    img = Image.fromarray((arr_01.clip(0, 1) * 255).astype('uint8'), mode='L')
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()


@app.post('/api/predict')
def api_predict(file: UploadFile = File(...), mode: str = Form('greedy'), early_exit: bool = Form(False)):
    if model is None:
        raise HTTPException(503, f"Model is not loaded ({load_error}). Check the model/ folder.")
    data = file.file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "The image is larger than 8 MB. Crop just the formula and try again.")
    try:
        img = Image.open(io.BytesIO(data))
        if img.width * img.height > MAX_PIXELS:
            raise HTTPException(413, "The image has too many pixels. Crop just the formula.")
        img.load()
    except UnidentifiedImageError:
        raise HTTPException(400, "The file is not an image. PNG, JPG, WebP and screenshots are supported.")

    cfg = info['config']
    sos, eos, max_len = tokenizer.sos, tokenizer.eos, info.get('max_gen_len', 250)

    def prepare(frac):
        return collate_images([preprocessor(prepare_user_image(img, ink_frac=frac))], cfg)

    with lock, torch.inference_mode():
        t0 = time.perf_counter()
        # 1) Formula size: try several margins and keep the one the model is most confident in
        #    (mean log-probability of the greedy output, including the end-of-sequence token).
        best = None
        for frac in INK_FRACTIONS:
            images, mask = prepare(frac)
            toks, _ = early_exit_generate(model, images, mask, sos, eos, use_early_exit=False, max_len=max_len)
            sc0 = score_sequence(model, images, mask, sos, eos, toks)
            score = sum(math.log(max(p, 1e-9)) for p in sc0['probs']) / len(sc0['probs'])
            if best is None or score > best[0]:
                best = (score, frac, images, mask, toks)
        _, frac, images, mask, greedy_tokens = best
        # 2) Decode in the selected mode at the best size
        if mode == 'beam':
            tokens, layers = beam_search(model, images, mask, sos, eos, beam_width=3, max_len=max_len), None
        elif early_exit:
            tokens, layers = early_exit_generate(model, images, mask, sos, eos, use_early_exit=True, max_len=max_len)
        else:
            tokens, layers = greedy_tokens, None
        gen_ms = (time.perf_counter() - t0) * 1000
        sc = score_sequence(model, images, mask, sos, eos, tokens)

    stride = feat_stride(cfg)
    Hf, Wf = sc['hw']
    vc = int(mask[0].view(Hf, Wf)[0].sum())
    maps = []
    for a in sc['alphas']:
        m = a.reshape(Hf, Wf)[:, :vc]
        m = m / max(float(m.max()), 1e-9)
        maps.append([[round(float(v), 3) for v in row] for row in m])
    shown = (images[0, 0] * 0.229 + 0.485).numpy()[:, :vc * stride]     # what the model sees (before normalisation)

    out_tokens = []
    for i, tid in enumerate(tokens):
        text, eow = tokenizer.piece(tid)
        out_tokens.append({'t': text, 'eow': eow, 'p': round(sc['probs'][i], 4),
                           'entropy': round(sc['entropies'][i], 4),
                           'layer': (layers[i] if layers else sc['exit_layers'][i])})
    result = {
        'latex': tokenizer.decode(tokens),
        'tokens': out_tokens,
        'end_prob': round(sc['probs'][-1], 4),
        'attention': {'rows': Hf, 'cols': vc, 'stride': stride, 'maps': maps[:len(tokens)]},
        'model_input': _png_data_url(shown),
        'mode': mode, 'early_exit': bool(early_exit and mode != 'beam'),
        'avg_layers': (sum(layers) / len(layers)) if layers else None,
        'ink_frac': frac,
        'ms': round(gen_ms),
    }
    if SAVE_DIR:                      # optional (for experiments): store the uploaded image and the model's answer
        import json
        os.makedirs(SAVE_DIR, exist_ok=True)
        stem = os.path.join(SAVE_DIR, f"{time.strftime('%Y%m%d_%H%M%S')}_{mode}_{os.path.splitext(file.filename or 'image')[0]}")
        open(stem + '.png', 'wb').write(data)
        with open(stem + '.json', 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False)
    return result
