"""NoteVision - console prediction of a printed music staff image.

Usage:
    python code/predict.py IMAGE [IMAGE ...] [--weights PATH] [--vocab PATH] [--json] [--pitch]

Prints the predicted sequence of agnostic tokens (symbol + staff position). If no clef is
predicted, the G clef is ASSUMED and this is reported explicitly (an assumption, not a detection).
Default files: artifacts/model/best.weights.h5 and artifacts/vocab_agnostic.csv
"""
import argparse
import csv
import json
import os
import re
import sys
import textwrap
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

IMG_HEIGHT = 128
REDUCTION = 8          # horizontal downsampling of the CNN
CTC_BLANK = 0          # class 0 = blank; token id i -> class i + 1
LSTM_UNITS = 128
# Training pads each image to (bucket boundary - 1); inference does the same for parity.
BOUNDS = list(range(600, 2201, 100)) + [2600, 3000, 3600]
MAX_TRAINED_WIDTH = 3600

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = ROOT / 'artifacts' / 'model' / 'best.weights.h5'
DEFAULT_VOCAB = ROOT / 'artifacts' / 'vocab_agnostic.csv'

ASSUMED_CLEF = 'clef.G-L2'
PITCHED = ('note', 'gracenote')
# Diatonic index (C=0 ... B=6, +7 per octave) of the note each clef marks: G4, F3, C4.
CLEF_NOTE = {'G': 4 * 7 + 4, 'F': 3 * 7 + 3, 'C': 4 * 7 + 0}
POSITION_RE = re.compile(r'^([LS])(-?\d+)$')


# ---------------------------------------------------------------- model
class CRNN(keras.Model):
    def __init__(self, num_classes, lstm_units=LSTM_UNITS, dropout=0.25, **kw):
        super().__init__(**kw)
        filters = (32, 64, 128, 256)
        pools = ((2, 2), (2, 2), (2, 2), (2, 1))  # width /2 three times -> x8
        self.convs = [layers.Conv2D(f, 3, padding='same', use_bias=False) for f in filters]
        self.bns = [layers.BatchNormalization() for _ in filters]
        self.pools = [layers.MaxPool2D(p) for p in pools]
        self.proj = layers.Dense(256, activation='relu')
        self.drop = layers.Dropout(dropout)
        self.rnn1 = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True))
        self.rnn2 = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True))
        self.classifier = layers.Dense(num_classes)

    def call(self, inputs, training=False):
        x, widths = inputs['image'], inputs['width']
        for conv, bn, pool in zip(self.convs, self.bns, self.pools):
            x = pool(tf.nn.relu(bn(conv(x), training=training)))
        x = tf.transpose(x, [0, 2, 1, 3])
        s = tf.shape(x)
        x = tf.reshape(x, [s[0], s[1], x.shape[2] * x.shape[3]])
        x = self.drop(self.proj(x), training=training)
        logit_len = widths // REDUCTION
        mask = tf.sequence_mask(logit_len, maxlen=s[1])
        x = self.rnn1(x, mask=mask, training=training)
        x = self.rnn2(self.drop(x, training=training), mask=mask, training=training)
        return self.classifier(self.drop(x, training=training)), logit_len


def load_vocab(path):
    with open(path, newline='', encoding='utf-8-sig') as f:
        tokens = [row['token'] for row in csv.DictReader(f)]
    if not tokens:
        raise ValueError(f'empty vocabulary: {path}')
    return tokens


def build_model(num_classes):
    model = CRNN(num_classes)
    model({'image': tf.zeros([1, IMG_HEIGHT, 256, 1]), 'width': tf.constant([256])})  # build weights
    return model


def load_model(weights, vocab):
    if not Path(weights).is_file():
        raise FileNotFoundError(f'weights not found: {weights} '
                                '(use --weights, or copy best.weights.h5 to artifacts/model/)')
    if not Path(vocab).is_file():
        raise FileNotFoundError(f'vocabulary not found: {vocab} '
                                '(use --vocab, or copy vocab_agnostic.csv to artifacts/model/)')
    tokens = load_vocab(vocab)
    model = build_model(len(tokens) + 1)
    try:
        model.load_weights(str(weights))
    except Exception as exc:
        raise RuntimeError('could not load the weights: check that the vocabulary matches the model '
                           'and that the TensorFlow/Keras versions match training '
                           f'(TF 2.21.0 / Keras 3.13.2). Details: {exc}') from exc
    return model, tokens


# ---------------------------------------------------------------- preprocessing
def read_rgb(path):
    """Decode PNG/JPG to uint8 RGB. True alpha is composited on white (not dropped)."""
    with Image.open(path) as im:
        im.load()
        if im.mode in ('RGBA', 'LA', 'PA'):
            rgba = im.convert('RGBA')
            background = Image.new('RGBA', rgba.size, (255, 255, 255, 255))
            im = Image.alpha_composite(background, rgba)
        return np.asarray(im.convert('RGB'))


def padded_width(width):
    for b in BOUNDS:
        if width < b:
            return b - 1
    return width


def preprocess(rgb):
    """Same steps as training: gray, height 128 keeping aspect, ink = 1, background = 0."""
    gray = tf.image.rgb_to_grayscale(tf.convert_to_tensor(rgb))  # uint8 [H, W, 1]
    h = tf.cast(tf.shape(gray)[0], tf.float32)
    w = tf.cast(tf.shape(gray)[1], tf.float32)
    new_w = int(tf.maximum(1, tf.cast(tf.round(w * IMG_HEIGHT / h), tf.int32)))
    gray = tf.image.resize(tf.cast(gray, tf.float32), (IMG_HEIGHT, new_w),
                           method='bilinear', antialias=True)
    gray = tf.clip_by_value(gray, 0.0, 255.0)
    image = (1.0 - gray / 255.0).numpy()
    pad_to = padded_width(new_w)
    if pad_to > new_w:
        image = np.pad(image, ((0, 0), (0, pad_to - new_w), (0, 0)))  # zeros = background
    return image, new_w


# ---------------------------------------------------------------- decoding
def greedy_decode(logits, lengths, blank=CTC_BLANK):
    ids = np.argmax(logits, axis=-1)
    out = []
    for seq, n in zip(ids, lengths):
        seq = seq[:n]
        changed = np.concatenate([[True], seq[1:] != seq[:-1]])
        seq = seq[changed]
        out.append((seq[seq != blank] - 1).tolist())
    return out


def predict_tokens(model, tokens, image, width):
    x = tf.convert_to_tensor(image[None], dtype=tf.float32)
    logits, logit_len = model({'image': x, 'width': tf.constant([width], dtype=tf.int32)},
                              training=False)
    ids = greedy_decode(logits.numpy(), logit_len.numpy())[0]
    return [tokens[i] for i in ids]


# ---------------------------------------------------------------- structure
def split_token(tok):
    fam, _, pos = tok.partition('-')
    kind, _, subtype = fam.partition('.')
    return {'token': tok, 'family': fam, 'kind': kind,
            'subtype': subtype or None, 'position': pos or None}


def staff_step(position):
    """Space below line 1 = 0; line k -> 2k-1; space k -> 2k. None if not parseable."""
    m = POSITION_RE.match(position or '')
    if m is None:
        return None
    n = int(m.group(2))
    return 2 * n - 1 if m.group(1) == 'L' else 2 * n


def pitch_name(position, clef_token):
    """Letter + octave from staff position and clef only (no key signature, no accidentals)."""
    clef = split_token(clef_token)
    step, clef_step = staff_step(position), staff_step(clef['position'])
    if step is None or clef_step is None or clef['subtype'] not in CLEF_NOTE:
        return None
    idx = CLEF_NOTE[clef['subtype']] + (step - clef_step)
    return f"{'CDEFGAB'[idx % 7]}{idx // 7}"


def build_events(tokens, with_pitch):
    events, current, basis = [], ASSUMED_CLEF, 'assumed'
    for tok in tokens:
        ev = split_token(tok)
        if ev['kind'] == 'clef':
            current, basis = tok, 'detected'
        elif with_pitch and ev['kind'] in PITCHED:
            ev['pitch'] = pitch_name(ev['position'], current)
            ev['clef_basis'] = basis
            ev['clef_used'] = current
        events.append(ev)
    return events


def clef_summary(tokens):
    clefs = [t for t in tokens if t.startswith('clef.')]
    if not clefs:
        return ({'status': 'assumed', 'token': ASSUMED_CLEF},
                [f'No clef was predicted; {ASSUMED_CLEF} (G clef) is ASSUMED for any interpretation. '
                 'This is an assumption, not a detection.'])
    assumptions = []
    first = tokens.index(clefs[0])
    if any(split_token(t)['kind'] in PITCHED for t in tokens[:first]):
        assumptions.append('Some notes appear before the first predicted clef; '
                           f'they are interpreted with the ASSUMED {ASSUMED_CLEF}.')
    return {'status': 'detected', 'token': clefs[0]}, assumptions


def predict_image(model, tokens, path, with_pitch=False):
    rgb = read_rgb(path)
    image, width = preprocess(rgb)
    pred = predict_tokens(model, tokens, image, width)
    clef, assumptions = clef_summary(pred)
    if width > MAX_TRAINED_WIDTH:
        assumptions.append(f'Image is {width} px wide after resizing to height {IMG_HEIGHT}, '
                           f'wider than any training image (< {MAX_TRAINED_WIDTH}); '
                           'the prediction may be unreliable.')
    return {'image': str(path), 'width_px': int(rgb.shape[1]), 'height_px': int(rgb.shape[0]),
            'tokens': pred, 'clef': clef, 'events': build_events(pred, with_pitch),
            'assumptions': assumptions}


# ---------------------------------------------------------------- output
def format_text(r, with_pitch):
    lines = [f"Image: {r['image']}  ({r['width_px']}x{r['height_px']} px)"]
    clef = r['clef']
    if clef['status'] == 'detected':
        lines.append(f"Clef: detected ({clef['token']})")
    else:
        lines.append(f"Clef: NOT detected -> {clef['token']} ASSUMED (assumption, not a detection)")
    lines.append(f"Predicted sequence ({len(r['tokens'])} tokens):")
    seq = ' '.join(r['tokens']) or '(empty prediction)'
    lines += textwrap.wrap(seq, width=100, initial_indent='  ', subsequent_indent='  ',
                           break_on_hyphens=False, break_long_words=False)
    if with_pitch:
        lines.append('Pitched events (letter+octave from staff position and clef only; '
                     'key signature and accidentals NOT applied):')
        for i, ev in enumerate(r['events']):
            if 'pitch' in ev:
                lines.append(f"  {i:>3}  {ev['token']:<30} {ev['pitch'] or '?':<4} "
                             f"[{ev['clef_basis']} clef: {ev['clef_used']}]")
    for a in r['assumptions']:
        lines.append(f'ASSUMPTION: {a}')
    return '\n'.join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description='Predict the music symbols of a printed staff image.')
    p.add_argument('images', nargs='+', help='PNG/JPG image(s) of a cropped printed staff')
    p.add_argument('--weights', type=Path, default=DEFAULT_WEIGHTS)
    p.add_argument('--vocab', type=Path, default=DEFAULT_VOCAB)
    p.add_argument('--json', action='store_true', help='print JSON instead of text')
    p.add_argument('--pitch', action='store_true',
                   help='add letter+octave for notes (clef only; no key signature/accidentals)')
    args = p.parse_args(argv)

    try:
        model, tokens = load_model(args.weights, args.vocab)
    except Exception as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 2

    results, failed = [], 0
    for path in args.images:
        try:
            results.append(predict_image(model, tokens, path, args.pitch))
        except Exception as exc:
            failed += 1
            print(f'error: {path}: {exc}', file=sys.stderr)

    if args.json:
        print(json.dumps(results[0] if len(results) == 1 else results, ensure_ascii=False, indent=2))
    else:
        for r in results:
            print(format_text(r, args.pitch))
            print()
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())