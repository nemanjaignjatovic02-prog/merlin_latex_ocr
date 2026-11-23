# -*- coding: utf-8 -*-
"""
merlin_model.py – shared model code for generating LaTeX from an image of a formula.

The same file is used by the Colab notebook (training, evaluation) and by the web app (inference),
so the model in the app is guaranteed to be the model described in the thesis.

Components are selected by the configuration (for the ablation study):
    preprocess    'aspect' (height 96 px, aspect ratio preserved) | 'square' (224×224)
    encoder_block 3 (stride 16, finer feature map) | 4 (stride 32)
    pos           '2d' (row, column) | '1d' (learned vector per region index)
    coverage      True | False
"""
import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms.functional as TF
from PIL import Image, ImageOps

DEFAULT_CONFIG = dict(
    preprocess='aspect', img_height=96, max_width=768, square_size=224,
    encoder_block=3, pos='2d', coverage=True,
    embed_size=256, hidden_size=512, num_layers=4,
)
SPECIAL_TOKENS = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def full_config(cfg=None):
    c = dict(DEFAULT_CONFIG)
    c.update(cfg or {})
    return c


def feat_stride(cfg):
    return 16 if cfg['encoder_block'] == 3 else 32


# =========================================================================================
#  Tokenizer
# =========================================================================================
def build_tokenizer(train_formulas_txt, vocab_size=5000):
    """BPE that reproduces the LaTeX source exactly: it splits only on spaces, and the </w> suffix marks
    the end of a LaTeX token. It is trained ONLY on formulas from the training set."""
    from tokenizers import Tokenizer, decoders
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import WhitespaceSplit
    tok = Tokenizer(BPE(unk_token="<UNK>", end_of_word_suffix="</w>"))
    tok.pre_tokenizer = WhitespaceSplit()
    tok.decoder = decoders.BPEDecoder(suffix="</w>")
    tok.train([train_formulas_txt], BpeTrainer(vocab_size=vocab_size, special_tokens=SPECIAL_TOKENS,
                                               end_of_word_suffix="</w>", show_progress=False))
    return tok


class TokenizerWrapper:
    def __init__(self, tok):
        self.tok = tok
        self.pad, self.sos, self.eos, self.unk = (tok.token_to_id(t) for t in SPECIAL_TOKENS)
        self.vocab_size = tok.get_vocab_size()

    @classmethod
    def from_file(cls, path):
        from tokenizers import Tokenizer
        return cls(Tokenizer.from_file(path))

    def encode(self, formula):
        return [self.sos] + self.tok.encode(formula).ids + [self.eos]

    def decode(self, ids):
        ids = [i for i in ids if i not in (self.pad, self.sos, self.eos)]
        return ' '.join(self.tok.decode(ids).split())

    def piece(self, i):
        """Text of one BPE token and whether it ends a LaTeX token (for per-token display)."""
        s = self.tok.id_to_token(i) or ''
        return s.replace('</w>', ''), s.endswith('</w>')


# =========================================================================================
#  Image preprocessing
# =========================================================================================
class ImagePreprocessor:
    """PIL image -> tensor [1, h, w] in the range 0..1 (1 = white)."""
    def __init__(self, cfg):
        self.cfg = full_config(cfg)

    def __call__(self, image):
        image = image.convert('L')
        if self.cfg['preprocess'] == 'square':
            s = self.cfg['square_size']
            return TF.to_tensor(image.resize((s, s), Image.BILINEAR))
        w, h = image.size
        s = min(self.cfg['img_height'] / h, self.cfg['max_width'] / w)
        image = image.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BILINEAR)
        return TF.to_tensor(image)


def prepare_user_image(image, ink_frac=None, pad_ratio=0.08):
    """User image (screenshot, photo) -> the look of the dataset images:
    greyscale, dark formula on white, cropped to the content with a white margin.

    ink_frac: share of the image height taken by the formula. In im2latex the formulas are surrounded by
    white space, so after scaling to 96 px they occupy only part of the height. A screenshot cropped
    "to the edge" would give much larger glyphs than in training, so white space is added around it
    (the app tries several values and keeps the one the model is most confident in)."""
    if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
        bg = Image.new('RGBA', image.size, (255, 255, 255, 255))
        image = Image.alpha_composite(bg, image.convert('RGBA'))
    g = np.asarray(image.convert('L'), dtype=np.float32)
    # dark background (e.g. dark mode) -> invert; the formula covers a minority of pixels, so the median = background
    if np.median(g) < 128:
        g = 255 - g
    # contrast stretching
    lo, hi = np.percentile(g, 1), np.percentile(g, 99)
    if hi - lo > 10:
        g = np.clip((g - lo) / (hi - lo) * 255, 0, 255)
    # ink/background threshold: Otsu's method (adapts to grey backgrounds and faint ink)
    hist = np.histogram(g, bins=256, range=(0, 256))[0].astype(np.float64)
    w0 = np.cumsum(hist); w1 = w0[-1] - w0
    m = np.cumsum(hist * np.arange(256)); mu0 = m / np.maximum(w0, 1); mu1 = (m[-1] - m) / np.maximum(w1, 1)
    thr = float(np.clip(np.argmax(w0 * w1 * (mu0 - mu1) ** 2), 100, 200))
    # remove frames/shadows along the border (rows and columns that are mostly dark)
    had_frame = False
    for _ in range(4):
        d = g < thr
        changed = False
        while g.shape[0] > 8 and d[0].mean() > 0.5:
            g, d, changed = g[1:], d[1:], True
        while g.shape[0] > 8 and d[-1].mean() > 0.5:
            g, d, changed = g[:-1], d[:-1], True
        while g.shape[1] > 8 and d[:, 0].mean() > 0.5:
            g, d, changed = g[:, 1:], d[:, 1:], True
        while g.shape[1] > 8 and d[:, -1].mean() > 0.5:
            g, d, changed = g[:, :-1], d[:, :-1], True
        had_frame |= changed
        if not changed:
            break
    if had_frame:                      # frame remnants (rounded corners) at the very edge -> white
        k = max(3, int(round(0.05 * min(g.shape))))
        g = g.copy()
        g[:k, :] = 255; g[-k:, :] = 255; g[:, :k] = 255; g[:, -k:] = 255
    ink = g < thr
    if ink.any():
        rows, cols = np.where(ink.any(1))[0], np.where(ink.any(0))[0]
        g = g[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
    h = g.shape[0]
    side = max(4, int(round(pad_ratio * h)))
    if ink_frac:
        v = max(side, int(round(h * (1.0 / ink_frac - 1.0) / 2)))
        g = np.pad(g, ((v, v), (side, side)), constant_values=255)
    else:
        g = np.pad(g, side, constant_values=255)
    return Image.fromarray(g.astype(np.uint8))


def collate_images(images, cfg):
    """List of tensors [1, h, w] -> (batch [B, 3, H, W] normalised for DenseNet, mask [B, H'*W']).
    Images are placed on a white canvas; the mask marks which regions are real image (not white padding)."""
    cfg = full_config(cfg)
    stride = feat_stride(cfg)
    H = math.ceil(max(im.shape[1] for im in images) / stride) * stride
    W = math.ceil(max(im.shape[2] for im in images) / stride) * stride
    canvas = torch.ones(len(images), 1, H, W)
    valid_cols = []
    for b, im in enumerate(images):
        h, w = im.shape[1:]
        top = (H - h) // 2
        canvas[b, :, top:top + h, :w] = im
        valid_cols.append(math.ceil(w / stride))
    batch = (canvas.repeat(1, 3, 1, 1) - IMAGENET_MEAN) / IMAGENET_STD
    Hf, Wf = H // stride, W // stride
    cols = torch.arange(Wf).view(1, 1, Wf).expand(len(images), Hf, Wf)
    mask = (cols < torch.tensor(valid_cols).view(-1, 1, 1)).reshape(len(images), -1)
    return batch, mask


# =========================================================================================
#  Model
# =========================================================================================
class PositionalEncoding2D(nn.Module):
    """Sinusoidal 2D encoding: half of the channels encode the row (y), half the column (x)."""
    def __init__(self, d_model, max_h=64, max_w=512):
        super().__init__()
        d = d_model // 2
        self.register_buffer('pe_y', self._pe_1d(max_h, d), persistent=False)
        self.register_buffer('pe_x', self._pe_1d(max_w, d), persistent=False)

    @staticmethod
    def _pe_1d(n, d):
        pos = torch.arange(n).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe = torch.zeros(n, d)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        return pe

    def forward(self, x):                                    # x: [B, C, H, W]
        _, _, H, W = x.shape
        pe = torch.cat([self.pe_y[:H].unsqueeze(1).expand(H, W, -1),
                        self.pe_x[:W].unsqueeze(0).expand(H, W, -1)], dim=-1)
        return x + pe.permute(2, 0, 1).unsqueeze(0).to(x.dtype)


class DenseNetEncoder(nn.Module):
    def __init__(self, cfg, pretrained=True):
        super().__init__()
        self.cfg = cfg
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        f = models.densenet121(weights=weights).features
        layers = [f.conv0, f.norm0, f.relu0, f.pool0, f.denseblock1, f.transition1,
                  f.denseblock2, f.transition2, f.denseblock3]
        if cfg['encoder_block'] == 3:
            self.norm = f.transition3.norm                   # output of block 3 (stride 16)
        else:
            layers += [f.transition3, f.denseblock4]
            self.norm = f.norm5                              # output of block 4 (stride 32)
        self.features = nn.Sequential(*layers)
        E = cfg['embed_size']
        self.conv = nn.Conv2d(1024, E, kernel_size=1)
        if cfg['pos'] == '2d':
            self.pos2d = PositionalEncoding2D(E)
        else:
            self.pos1d = nn.Parameter(torch.randn(1, 2048, E) * 0.02)

    def forward(self, images):
        x = F.relu(self.norm(self.features(images)))
        x = F.relu(self.conv(x))
        if self.cfg['pos'] == '2d':
            x = self.pos2d(x)
        _, _, H, W = x.shape
        x = x.flatten(2).permute(0, 2, 1)                    # [B, H*W, E]
        if self.cfg['pos'] == '1d':
            x = x + self.pos1d[:, :x.size(1), :].to(x.dtype)
        return x, (H, W)


class CoverageAttention(nn.Module):
    def __init__(self, hidden_size, embed_size, use_coverage=True, cov_channels=32, cov_kernel=(3, 7)):
        super().__init__()
        self.use_coverage = use_coverage
        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.kv_linear = nn.Linear(embed_size, hidden_size * 2)
        self.scale = hidden_size ** -0.5
        if use_coverage:
            self.cov_conv = nn.Conv2d(1, cov_channels, cov_kernel, padding=(cov_kernel[0] // 2, cov_kernel[1] // 2))
            self.cov_proj = nn.Linear(cov_channels, hidden_size // 4)
            self.cov_score = nn.Linear(hidden_size // 4, 1, bias=False)
            nn.init.zeros_(self.cov_score.weight)

    def precompute(self, features):
        key, value = self.kv_linear(features).chunk(2, dim=-1)
        return key, value

    def forward(self, hidden, key, value, coverage, hw, mask=None):
        q = self.q_linear(hidden).unsqueeze(1)                                   # [B, 1, Hd]
        scores = (torch.bmm(q, key.transpose(1, 2)).squeeze(1) * self.scale).float()   # [B, N]
        if self.use_coverage:
            H, W = hw
            cov = self.cov_conv(coverage.view(-1, 1, H, W).to(q.dtype)).flatten(2).transpose(1, 2)
            scores = scores + self.cov_score(torch.tanh(self.cov_proj(cov))).squeeze(-1).float()
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))
        alpha = F.softmax(scores, dim=-1)
        context = torch.bmm(alpha.unsqueeze(1).to(value.dtype), value).squeeze(1)
        return context, alpha


class GRUDecoder(nn.Module):
    def __init__(self, cfg, vocab_size):
        super().__init__()
        E, Hd, L = cfg['embed_size'], cfg['hidden_size'], cfg['num_layers']
        self.num_layers = L
        self.embedding = nn.Embedding(vocab_size, E)
        self.attention = CoverageAttention(Hd, E, cfg['coverage'])
        self.init_hidden = nn.Linear(E, Hd)
        self.gru_layers = nn.ModuleList([nn.GRUCell(E + Hd if i == 0 else Hd, Hd) for i in range(L)])
        self.classifier = nn.Linear(Hd, vocab_size)

    def init_state(self, features, mask):
        m = mask.unsqueeze(-1).to(features.dtype)
        h0 = self.init_hidden((features * m).sum(1) / m.sum(1).clamp(min=1))
        coverage = torch.zeros(features.size(0), features.size(1), device=features.device)
        return [h0] * self.num_layers, coverage

    def forward(self, features, hw, mask, captions, ss_prob=0.0):
        """Training pass. ss_prob > 0 enables scheduled sampling: with that probability the input token
        is the model's prediction from the previous step instead of the ground-truth token."""
        key, value = self.attention.precompute(features)
        hidden, coverage = self.init_state(features, mask)
        outs = {i: [] for i in range(self.num_layers)}
        prev_pred = None
        for t in range(captions.size(1)):
            token = captions[:, t]
            if ss_prob > 0 and prev_pred is not None:
                use = torch.rand(token.size(0), device=token.device) < ss_prob
                token = torch.where(use, prev_pred, token)
            context, alpha = self.attention(hidden[-1], key, value, coverage, hw, mask)
            coverage = coverage + alpha
            x = torch.cat([self.embedding(token), context], dim=1)
            for i, cell in enumerate(self.gru_layers):
                hidden[i] = cell(x, hidden[i])
                x = hidden[i]
                outs[i].append(self.classifier(x))
            prev_pred = outs[self.num_layers - 1][-1].detach().argmax(-1)
        return {i: torch.stack(v, dim=1) for i, v in outs.items()}


class Im2LatexModel(nn.Module):
    def __init__(self, cfg, vocab_size, pretrained=True):
        super().__init__()
        self.cfg = full_config(cfg)
        self.encoder = DenseNetEncoder(self.cfg, pretrained)
        self.decoder = GRUDecoder(self.cfg, vocab_size)

    def forward(self, images, mask, captions, ss_prob=0.0):
        features, hw = self.encoder(images)
        return self.decoder(features, hw, mask, captions, ss_prob)


class KDLoss(nn.Module):
    """CE with the ground-truth token + KL towards the teacher's softened distribution; <PAD> positions are ignored."""
    def __init__(self, temperature=2.0, alpha=0.4, pad_idx=0):
        super().__init__()
        self.T, self.alpha, self.pad_idx = temperature, alpha, pad_idx

    def forward(self, student_logits, teacher_logits, targets):
        V = student_logits.size(-1)
        y = targets.reshape(-1)
        valid = y != self.pad_idx
        s = student_logits.reshape(-1, V)[valid].float()
        t = teacher_logits.reshape(-1, V)[valid].float()
        loss_ce = F.cross_entropy(s, y[valid])
        loss_kd = F.kl_div(F.log_softmax(s / self.T, -1), F.softmax(t / self.T, -1),
                           reduction='batchmean') * self.T ** 2
        return (1.0 - self.alpha) * loss_ce + self.alpha * loss_kd


def total_loss(model, outputs, targets, pad_idx, kd_loss, student_weight=0.2):
    L = model.decoder.num_layers
    V = outputs[L - 1].size(-1)
    teacher = outputs[L - 1].detach()
    ce_top = F.cross_entropy(outputs[L - 1].reshape(-1, V).float(), targets.reshape(-1), ignore_index=pad_idx)
    loss = ce_top
    for i in range(L - 1):
        loss = loss + student_weight * kd_loss(outputs[i], teacher, targets)
    return loss, ce_top


# =========================================================================================
#  Generation
# =========================================================================================
def _step(dec, token, hidden, key, value, coverage, hw, mask):
    context, alpha = dec.attention(hidden[-1], key, value, coverage, hw, mask)
    x = torch.cat([dec.embedding(token), context], dim=1)
    new_hidden = []
    for i, cell in enumerate(dec.gru_layers):
        x = cell(x, hidden[i])
        new_hidden.append(x)
    return dec.classifier(x), new_hidden, coverage + alpha, alpha


def _cut(seq, eos):
    out = []
    for t in seq:
        if t == eos:
            break
        out.append(t)
    return out


@torch.no_grad()
def greedy_batch(model, images, mask, sos, eos, max_len=250):
    """Greedy decoding through all 4 layers for a whole batch at once (fast evaluation of the full test set)."""
    model.eval()
    dec = model.decoder
    features, hw = model.encoder(images)
    key, value = dec.attention.precompute(features)
    hidden, coverage = dec.init_state(features, mask)
    B = images.size(0)
    token = torch.full((B,), sos, dtype=torch.long, device=images.device)
    out = torch.full((B, max_len), eos, dtype=torch.long, device=images.device)
    done = torch.zeros(B, dtype=torch.bool, device=images.device)
    for t in range(max_len):
        logits, hidden, coverage, _ = _step(dec, token, hidden, key, value, coverage, hw, mask)
        token = logits.argmax(-1)
        out[:, t] = torch.where(done, torch.full_like(token, eos), token)
        done |= token == eos
        if done.all():
            break
    return [_cut(s, eos) for s in out.tolist()]


@torch.no_grad()
def beam_search(model, image, mask, sos, eos, beam_width=3, max_len=250):
    """Beam search for one image; all beams are computed together as a batch, each with its own
    coverage map. The sequence with the best length-normalised log-probability is returned."""
    model.eval()
    dec, k = model.decoder, beam_width
    features, hw = model.encoder(image)
    key, value = dec.attention.precompute(features)
    hidden, coverage = dec.init_state(features, mask)
    key, value, mask_k = key.expand(k, -1, -1), value.expand(k, -1, -1), mask.expand(k, -1)
    hidden = [h.expand(k, -1).contiguous() for h in hidden]
    coverage = coverage.expand(k, -1).contiguous()
    seqs = [[sos] for _ in range(k)]
    scores = torch.full((k,), float('-inf'), device=image.device)
    scores[0] = 0.0
    completed = []
    for _ in range(max_len):
        token = torch.tensor([s[-1] for s in seqs], device=image.device)
        logits, new_hidden, new_cov, _ = _step(dec, token, hidden, key, value, coverage, hw, mask_k)
        logp = F.log_softmax(logits.float(), dim=-1)
        V = logp.size(1)
        top_s, top_i = (scores.unsqueeze(1) + logp).view(-1).topk(2 * k)
        nxt = []
        for s, idx in zip(top_s.tolist(), top_i.tolist()):
            if s == float('-inf'):
                continue
            b, w = divmod(idx, V)
            if w == eos:
                completed.append((s / (len(seqs[b]) + 1), seqs[b][1:]))
            elif len(nxt) < k:
                nxt.append((s, b, w))
        if len(completed) >= k or not nxt:
            break
        sel = torch.tensor([b for _, b, _ in nxt], device=image.device)
        seqs = [seqs[b] + [w] for _, b, w in nxt]
        scores = torch.tensor([s for s, _, _ in nxt], device=image.device)
        hidden = [h[sel] for h in new_hidden]
        coverage = new_cov[sel]
        n = len(nxt)
        key, value, mask_k = key[:n], value[:n], mask_k[:n]
    if not completed:
        completed = [(s / len(q), q[1:]) for s, q in zip(scores.tolist(), seqs)]
    return max(completed, key=lambda x: x[0])[1]


def dynamic_threshold(step):
    return max(0.4, 0.8 - 0.005 * step)


@torch.no_grad()
def early_exit_generate(model, image, mask, sos, eos, threshold='dynamic', use_early_exit=True,
                        max_len=250, repetition_penalty=1.0):
    """Greedy generation with early exit. After each of the first 3 layers the entropy is computed;
    if it is below the threshold, the token is emitted immediately and the skipped upper layers receive a COPY
    of the exit layer's state (so the next step and the attention do not use a stale state).
    Returns (tokens, number of layers used per token)."""
    model.eval()
    dec = model.decoder
    L = dec.num_layers
    features, hw = model.encoder(image)
    key, value = dec.attention.precompute(features)
    hidden, coverage = dec.init_state(features, mask)
    token = torch.tensor([sos], device=image.device)
    out, layers_used = [], []
    for step in range(max_len):
        thr = dynamic_threshold(step) if threshold == 'dynamic' else threshold
        context, alpha = dec.attention(hidden[-1], key, value, coverage, hw, mask)
        coverage = coverage + alpha
        x = torch.cat([dec.embedding(token), context], dim=1)
        exit_layer, logits = L - 1, None
        for i, cell in enumerate(dec.gru_layers):
            hidden[i] = cell(x, hidden[i])
            x = hidden[i]
            if use_early_exit and i < L - 1:
                lg = dec.classifier(x)
                p = F.softmax(lg.float(), dim=-1)
                if -(p * torch.log(p + 1e-9)).sum().item() < thr:
                    exit_layer, logits = i, lg
                    break
        if logits is None:
            logits = dec.classifier(hidden[-1])
        for j in range(exit_layer + 1, L):
            hidden[j] = hidden[exit_layer]
        if repetition_penalty > 1.0 and out:
            logits = logits.clone()
            idx = torch.tensor(sorted(set(out)), device=logits.device)
            v = logits[0, idx]
            logits[0, idx] = torch.where(v > 0, v / repetition_penalty, v * repetition_penalty)  # sign-aware repetition penalty
        layers_used.append(exit_layer + 1)
        tok = logits.argmax(-1).item()
        if tok == eos:
            break
        out.append(tok)
        token = torch.tensor([tok], device=image.device)
    return out, layers_used


@torch.no_grad()
def score_sequence(model, image, mask, sos, eos, tokens):
    """Forced pass over given tokens (e.g. a beam search result): for every token returns its probability,
    entropy, the layer at which early exit would emit it, and the attention map. Used for display in the app."""
    model.eval()
    dec = model.decoder
    L = dec.num_layers
    features, hw = model.encoder(image)
    key, value = dec.attention.precompute(features)
    hidden, coverage = dec.init_state(features, mask)
    seq = [sos] + list(tokens) + [eos]
    probs, entropies, exit_layers, alphas = [], [], [], []
    for t in range(len(seq) - 1):
        token = torch.tensor([seq[t]], device=image.device)
        context, alpha = dec.attention(hidden[-1], key, value, coverage, hw, mask)
        coverage = coverage + alpha
        x = torch.cat([dec.embedding(token), context], dim=1)
        exit_at = L
        for i, cell in enumerate(dec.gru_layers):
            hidden[i] = cell(x, hidden[i])
            x = hidden[i]
            if i < L - 1 and exit_at == L:
                p_i = F.softmax(dec.classifier(x).float(), -1)[0]
                if -(p_i * torch.log(p_i + 1e-9)).sum().item() < dynamic_threshold(t):
                    exit_at = i + 1
        p = F.softmax(dec.classifier(hidden[-1]).float(), -1)[0]
        probs.append(float(p[seq[t + 1]]))
        entropies.append(float(-(p * torch.log(p + 1e-9)).sum()))
        exit_layers.append(exit_at)
        alphas.append(alpha[0].float().cpu().numpy())
    return dict(probs=probs, entropies=entropies, exit_layers=exit_layers, alphas=alphas, hw=hw)


# =========================================================================================
#  Loading / saving for the app
# =========================================================================================
def save_for_inference(model, tokenizer, out_dir, extra_info=None):
    """Saves everything the app needs: weights, configuration, tokenizer and (optionally) results."""
    import json, os
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, 'model.pt'))
    tokenizer.tok.save(os.path.join(out_dir, 'tokenizer.json'))
    info = {'config': model.cfg, 'vocab_size': tokenizer.vocab_size}
    info.update(extra_info or {})
    with open(os.path.join(out_dir, 'model_info.json'), 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=1)


def load_for_inference(model_dir, device='cpu'):
    import json, os
    with open(os.path.join(model_dir, 'model_info.json'), encoding='utf-8') as f:
        info = json.load(f)
    tokenizer = TokenizerWrapper.from_file(os.path.join(model_dir, 'tokenizer.json'))
    model = Im2LatexModel(info['config'], tokenizer.vocab_size, pretrained=False)
    state = torch.load(os.path.join(model_dir, 'model.pt'), map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model, tokenizer, info


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
