"""Evaluate a pipeline on a directory of images and canonical Agent YOLO labels."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from valorant_portrait_kit.pipeline import Pipeline
from valorant_portrait_kit.metrics import match_detections


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--images', type=Path, required=True)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    pipeline = Pipeline(args.bundle)
    rows = []
    for image in sorted(args.images.rglob('*')):
        if image.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.webp'}:
            continue
        label = args.labels / image.relative_to(args.images).with_suffix('.txt')
        boxes = [list(map(float, line.split())) for line in label.read_text().splitlines() if line.strip()]
        im = cv2.imdecode(np.fromfile(str(image), np.uint8), cv2.IMREAD_COLOR)
        if im is None:
            raise ValueError(f'Cannot decode {image}')
        result = match_detections(pipeline.predict(im)['detections'], boxes, im.shape[1], im.shape[0], pipeline.names)
        rows.append({'image': str(image.relative_to(args.images)), **result})
    if not rows:
        parser.error('No images found')
    correct = sum(r['correct_name_and_box'] for r in rows)
    predicted = sum(r['predictions'] for r in rows)
    truth = sum(r['ground_truth'] for r in rows)
    output = {'images': len(rows), 'truth': truth, 'predictions': predicted, 'correct_name_and_box': correct,
              'precision': correct / predicted if predicted else 0, 'recall': correct / truth if truth else 0, 'per_image': rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in output.items() if k != 'per_image'}, indent=2))


if __name__ == '__main__':
    main()
