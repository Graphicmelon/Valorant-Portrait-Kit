"""Build a portable template bank from named RGBA PNG portraits."""
import argparse
import json
from pathlib import Path
from valorant_portrait_kit.templates import TemplateMatcher


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--portraits', type=Path, required=True)
    parser.add_argument('--classes', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    names = json.loads(args.classes.read_text(encoding='utf-8'))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    TemplateMatcher.build(args.portraits, names, args.output)


if __name__ == '__main__':
    main()
