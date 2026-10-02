"""Rebuild labeled portrait tiles with fixed RGBA templates and appearance noise.

Manifest: JSON list of {image, split, boxes}; boxes are YOLO Agent labels.
Only train images supply synthesis backgrounds; dev images are re-encoded
into separate disposable validation staging. Test images are never decoded.
"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--portraits', type=Path, required=True)
    parser.add_argument('--classes', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--count', type=int, default=180)
    parser.add_argument('--seed', type=int, default=20261001)
    args = parser.parse_args()
    records = json.loads(args.manifest.read_text(encoding='utf-8'))
    names = json.loads(args.classes.read_text(encoding='utf-8'))
    train = [r for r in records if r['split'] == 'train']
    dev = [r for r in records if r['split'] == 'dev']
    if not train or not dev or args.count < 1:
        parser.error('Nonempty train/dev splits and positive count are required')
    # Relative image paths are resolved against the manifest location.
    for r in records:
        r['image'] = str((args.manifest.parent / r['image']).resolve())
    if len({r['image'] for r in records}) != len(records):
        parser.error('An image occurs more than once in the split manifest')
    out = args.output.resolve()
    if out.exists():
        parser.error('Output must be a new directory to preserve existing data')
    for split in ('train', 'dev'):
        for kind in ('images', 'labels'):
            (out / kind / split).mkdir(parents=True)
    templates = {}
    for n in names:
        with Image.open(args.portraits / (n + '.png')) as im:
            templates[n] = im.convert('RGBA')
    rng = np.random.default_rng(args.seed)
    paths, provenance = [], []

    def read(record):
        im = cv2.imdecode(np.fromfile(record['image'], np.uint8), cv2.IMREAD_COLOR)
        if im is None:
            raise ValueError(f"Cannot decode {record['image']}")
        return im

    def save(im, stem, split, boxes):
        path = out / 'images' / split / (stem + '.jpg')
        ok, data = cv2.imencode('.jpg', im, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            raise RuntimeError('JPEG encoding failed')
        data.tofile(str(path))
        (out / 'labels' / split / (stem + '.txt')).write_text(
            ''.join(f'0 {x:.8f} {y:.8f} {w:.8f} {h:.8f}\n' for c, x, y, w, h in boxes))
        return str(path)

    for i, r in enumerate(train):
        paths.extend([save(read(r), f'real_{i:04}', 'train', r['boxes'])] * 3)
    for i, r in enumerate(dev):
        save(read(r), f'dev_{i:04}', 'dev', r['boxes'])
    total = sum(len(train[i % len(train)]['boxes']) for i in range(args.count))
    identities = np.resize(np.arange(len(names)), total)
    rng.shuffle(identities)
    offset = 0
    for i in range(args.count):
        r = train[i % len(train)]
        im = read(r)
        specs = []
        for _, cx, cy, bw, bh in r['boxes']:
            h, w = im.shape[:2]
            x1, y1, x2, y2 = [round(v) for v in
                ((cx-bw/2)*w, (cy-bh/2)*h, (cx+bw/2)*w, (cy+bh/2)*h)]
            if not (0 <= x1 < x2 < w-2 and 0 <= y1 < y2 <= h):
                raise ValueError('A tile has no valid neighboring backplate strip')
            strip = im[y1:y2, x2+2:min(x2+15, w)]
            plate = np.repeat(np.median(strip, axis=1)[:, None, :], x2-x1, axis=1)
            plate = np.clip(plate + rng.uniform(-14, 14, (1, 1, 3)), 0, 255)
            cid = int(identities[offset]); offset += 1
            rgba = np.asarray(templates[names[cid]].resize((x2-x1, y2-y1), Image.Resampling.LANCZOS), np.float32)
            rgb = rgba[:, :, :3][:, :, ::-1]
            gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)[:, :, None]
            dead = bool(rng.random() < .5)
            saturation = float(rng.uniform(.30, .85) if dead else rng.uniform(.85, 1.08))
            brightness = float(rng.uniform(.32, .65) if dead else rng.uniform(.82, 1.10))
            rgb = np.clip((gray + (rgb-gray)*saturation)*brightness, 0, 255)
            alpha = rgba[:, :, 3:4]/255
            im[y1:y2, x1:x2] = np.clip(rgb*alpha + plate*(1-alpha), 0, 255).astype(np.uint8)
            specs.append({'name': names[cid], 'dead': dead, 'saturation': saturation, 'brightness': brightness})
        scale = float(rng.choice([1., 1., .85, .7, .55]))
        if scale != 1:
            im = cv2.resize(im, (round(im.shape[1]*scale), round(im.shape[0]*scale)), interpolation=cv2.INTER_AREA)
        quality = int(rng.integers(45, 96))
        ok, data = cv2.imencode('.jpg', im, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise RuntimeError('JPEG degradation failed')
        paths.append(save(cv2.imdecode(data, cv2.IMREAD_COLOR), f'synthetic_{i:04}', 'train', r['boxes']))
        provenance.append({'background': r['image'], 'scale': scale, 'quality': quality, 'portraits': specs})
    (out / 'train.txt').write_text('\n'.join(paths) + '\n')
    (out / 'dataset.yaml').write_text(yaml.safe_dump(
        {'path': str(out), 'train': 'train.txt', 'val': 'images/dev', 'names': {0: 'avatar'}}, sort_keys=False))
    (out / 'synthesis.json').write_text(json.dumps({'seed': args.seed, 'outputs': provenance}, indent=2))
    print(out / 'dataset.yaml')


if __name__ == '__main__':
    main()
