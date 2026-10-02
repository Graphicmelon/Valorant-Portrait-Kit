"""Train YOLO26n locally with conservative RAM/VRAM use and epoch telemetry."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
(ROOT / '.cache' / 'ultralytics').mkdir(parents=True, exist_ok=True)
os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / '.cache' / 'ultralytics'))
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
os.environ.setdefault('YOLO_AUTOINSTALL', 'false')

import argparse
import json
import subprocess
import time

import psutil
import torch
import ultralytics
import yaml
from ultralytics import YOLO

REPORTS = ROOT/'reports'


def telemetry(trainer):
    stats = {'epoch': trainer.epoch + 1, 'time': time.time(),
        'vram_allocated_mb': torch.cuda.memory_allocated()/2**20,
        'vram_reserved_mb': torch.cuda.memory_reserved()/2**20,
        'vram_peak_mb': torch.cuda.max_memory_allocated()/2**20,
        'process_ram_mb': psutil.Process().memory_info().rss/2**20,
        'available_ram_gb': psutil.virtual_memory().available/2**30,
        'metrics': {str(k):float(v) for k,v in trainer.metrics.items()}}
    try:
        raw = subprocess.check_output(['nvidia-smi', '--query-gpu=temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw', '--format=csv,noheader,nounits'], text=True).strip()
        stats['gpu'] = dict(zip(['temperature_c','utilization_pct','memory_used_mb','memory_total_mb','power_w'], [float(x.strip()) for x in raw.split(',')]))
    except Exception as exc:
        stats['telemetry_warning'] = str(exc)
    with (REPORTS/'training_telemetry.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps(stats)+'\n')
    (REPORTS/'training_status.json').write_text(json.dumps(stats, indent=2), encoding='utf-8')
    # A brief cooldown at sustained high temperatures protects the shared laptop.
    if stats.get('gpu', {}).get('temperature_c', 0) >= 84:
        print('GPU temperature >=84C: allowing a 10-second cooldown.', flush=True)
        time.sleep(10)


def main():
    global REPORTS
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='avatar_detector')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch', type=int)
    parser.add_argument('--weights', default='yolo26n.pt')
    parser.add_argument('--config', type=Path, default=ROOT/'configs'/'train.yaml')
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--report-dir', type=Path, default=ROOT/'reports')
    args = parser.parse_args()
    REPORTS = args.report_dir.resolve()
    REPORTS.mkdir(parents=True,exist_ok=True)
    assert torch.cuda.is_available(), 'CUDA is unavailable; refusing accidental CPU training.'
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(0.60, device=0)
    config = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    if args.epochs:
        config['epochs'] = args.epochs
    if args.batch:
        config['batch'] = args.batch
    local_data = args.data.resolve()
    environment = {'torch': torch.__version__, 'ultralytics': ultralytics.__version__,
        'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0),
        'ram_gb': psutil.virtual_memory().total/2**30, 'config': config,
        'base_weights': args.weights, 'dataset': str(local_data)}
    (REPORTS/'training_environment.json').write_text(json.dumps(environment, indent=2), encoding='utf-8')
    print(json.dumps(environment, indent=2), flush=True)
    model = YOLO(args.weights)
    model.add_callback('on_fit_epoch_end', telemetry)
    model.train(data=str(local_data), project=str(ROOT/'runs'), name=args.name, exist_ok=False, **config)
    print('Training finished. Best weights:', model.trainer.best, flush=True)


if __name__ == '__main__':
    main()
