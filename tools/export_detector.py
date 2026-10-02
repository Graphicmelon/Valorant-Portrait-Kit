"""Export a single-class YOLO26 checkpoint to static FP32 ONNX."""
import argparse
import os
import shutil
from pathlib import Path
os.environ.setdefault('YOLO_AUTOINSTALL', 'false')
_config = Path(__file__).resolve().parents[1] / '.cache' / 'ultralytics'
_config.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('YOLO_CONFIG_DIR', str(_config))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--size', type=int, default=800)
    args = parser.parse_args()
    import onnx
    from ultralytics import YOLO
    model = YOLO(args.weights)
    if model.names != {0: 'avatar'}:
        parser.error('The checkpoint must use the single class avatar')
    exported = Path(model.export(
        format='onnx', imgsz=args.size, batch=1, device='cpu', dynamic=False,
        simplify=True, opset=17, nms=True, max_det=30, quantize=32,
        conf=0.01, iou=0.5, agnostic_nms=True))
    onnx.checker.check_model(str(exported))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if exported.resolve() != args.output.resolve():
        shutil.copy2(exported, args.output)
    print(args.output)


if __name__ == '__main__':
    main()
