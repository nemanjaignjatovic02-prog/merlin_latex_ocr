"""Smoke test: checks that merlin_model.py and app.py work together (randomly initialised model, no training).

Run (locally or in Colab):
    pip install -r requirements.txt torch torchvision httpx
    python test_app.py
If it prints "ALL CHECKS PASSED", the app is ready; only the real files from the training notebook
still have to be placed in model/.
"""
import glob
import json
import os
import sys
import tempfile

import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from merlin_model import (DEFAULT_CONFIG, ImagePreprocessor, Im2LatexModel, KDLoss, TokenizerWrapper, beam_search,
                       build_tokenizer, collate_images, early_exit_generate, greedy_batch, load_for_inference,
                       prepare_user_image, save_for_inference, score_sequence, total_loss)

formulas = list(json.load(open(os.path.join(HERE, 'static', 'examples', 'formulas.json'))).values())
tmp = tempfile.mkdtemp()
with open(os.path.join(tmp, 'f.txt'), 'w') as f:
    f.write('\n'.join(formulas * 20))
tok = TokenizerWrapper(build_tokenizer(os.path.join(tmp, 'f.txt'), vocab_size=300))
assert all(tok.decode(tok.encode(x)) == x for x in formulas), "tokenizer does not round-trip the text"

for cfg_over in ({}, {'preprocess': 'square'}, {'encoder_block': 4}, {'pos': '1d'}, {'coverage': False}):
    cfg = dict(DEFAULT_CONFIG, **cfg_over)
    model = Im2LatexModel(cfg, tok.vocab_size, pretrained=False)
    pre = ImagePreprocessor(cfg)
    imgs = [pre(prepare_user_image(Image.open(p))) for p in sorted(glob.glob(os.path.join(HERE, 'static/examples/*.png')))]
    images, mask = collate_images(imgs, cfg)
    toks = torch.nn.utils.rnn.pad_sequence([torch.tensor(tok.encode(x)) for x in formulas], batch_first=True,
                                           padding_value=tok.pad)
    out = model(images, mask, toks[:, :-1], ss_prob=0.2)
    loss, ce = total_loss(model, out, toks[:, 1:], tok.pad, KDLoss(pad_idx=tok.pad))
    loss.backward()
    assert torch.isfinite(loss), "loss is not finite"
    seqs = greedy_batch(model, images, mask, tok.sos, tok.eos, max_len=12)
    one_img, one_mask = collate_images(imgs[:1], cfg)
    b = beam_search(model, one_img, one_mask, tok.sos, tok.eos, 3, 12)
    e, layers = early_exit_generate(model, one_img, one_mask, tok.sos, tok.eos, use_early_exit=True, max_len=12,
                                    repetition_penalty=1.2)
    sc = score_sequence(model, one_img, one_mask, tok.sos, tok.eos, b)
    assert len(sc['probs']) == len(b) + 1 and len(sc['alphas']) == len(b) + 1
    print(f"OK {cfg_over or 'baseline'}: loss={loss.item():.3f}, greedy={len(seqs)}, beam={len(b)} tok, "
          f"early exit={len(e)} tok, map {sc['hw']}")

# --- save/load and the API -------------------------------------------------------------
model = Im2LatexModel(DEFAULT_CONFIG, tok.vocab_size, pretrained=False)
mdir = os.path.join(tmp, 'model')
save_for_inference(model, tok, mdir, {'metrics': {'test_em_full': 0.0}, 'max_gen_len': 20})
m2, t2, info = load_for_inference(mdir)
os.environ['MODEL_DIR'] = mdir
from fastapi.testclient import TestClient
import app as webapp
c = TestClient(webapp.app)
assert c.get('/').status_code == 200
inf = c.get('/api/info').json()
assert inf['ready'], inf
for mode, ee in (('greedy', 'false'), ('greedy', 'true'), ('beam', 'false')):
    with open(os.path.join(HERE, 'static/examples/example1.png'), 'rb') as f:
        r = c.post('/api/predict', files={'file': ('a.png', f, 'image/png')}, data={'mode': mode, 'early_exit': ee})
    assert r.status_code == 200, r.text
    d = r.json()
    assert len(d['attention']['maps']) == len(d['tokens'])
    assert d['model_input'].startswith('data:image/png;base64,')
    print(f"OK API {mode} early_exit={ee}: {len(d['tokens'])} tokens, {d['ms']} ms, grid "
          f"{d['attention']['rows']}×{d['attention']['cols']}")
r = c.post('/api/predict', files={'file': ('a.txt', b'not an image', 'text/plain')})
assert r.status_code == 400
print("ALL CHECKS PASSED")
