"""Generate a deterministic multi-photographer scale fixture (no external images)."""
from __future__ import annotations
import argparse
import hashlib
import os
import random
from datetime import datetime, timedelta
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter

COLORS = [
    ('#3c8dbd', '#f1d6a7'), ('#bd6650', '#49392b'),
    ('#70ab71', '#e8ec91'), ('#6a63ae', '#e2bfd8'),
    ('#d8a43a', '#383f48'), ('#4f9e96', '#f2d2a2'),
    ('#943e72', '#e7c18b'), ('#607baf', '#d7e2e9'),
]
PEOPLE = ('amy', 'ben', 'chris', 'dana')

def image_for(moment: int, index: int) -> Image.Image:
    rng = random.Random(moment * 10000 + index)
    base, accent = COLORS[moment]
    image = Image.new('RGB', (256, 192), base)
    draw = ImageDraw.Draw(image)
    # Each scene shares a motif while positions and details vary per photo.
    for j in range(5):
        x, y = rng.randint(10, 180), rng.randint(10, 120)
        width, height = rng.randint(20, 65), rng.randint(20, 60)
        if moment % 3 == 0:
            draw.ellipse((x, y, x+width, y+height), fill=accent, outline='white', width=2)
        elif moment % 3 == 1:
            draw.rectangle((x, y, x+width, y+height), fill=accent, outline='white', width=2)
        else:
            draw.polygon([(x, y+height), (x+width//2, y), (x+width, y+height)], fill=accent, outline='white')
    draw.text((8, 168), f'Scene {moment+1} frame {index+1}', fill='white')
    if index % 13 == 12:
        image = image.filter(ImageFilter.GaussianBlur(3))
    return image

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    labels = ['filename,moment_id,keep']
    start = datetime(2026, 9, 28, 9)
    for moment in range(8):
        originals: list[Path] = []
        for index in range(45):
            person = PEOPLE[(moment + index) % 4]
            folder = root/person
            folder.mkdir(exist_ok=True)
            path = folder/f'm{moment+1:02d}_p{index+1:03d}.png'
            image_for(moment, index).save(path)
            taken = start + timedelta(minutes=moment*100+index)
            stamp = taken.timestamp()
            os.utime(path, (stamp, stamp))
            originals.append(path)
            labels.append(f'{path.name},moment_{moment+1:02d},{int(index==0)}')
        for copy in range(5):
            source = originals[copy]
            person = PEOPLE[(moment+copy+1) % 4]
            path = root/person/f'm{moment+1:02d}_copy{copy+1:02d}.png'
            path.write_bytes(source.read_bytes())
            taken = start + timedelta(minutes=moment*100+copy+1)
            stamp = taken.timestamp()
            os.utime(path, (stamp, stamp))
            labels.append(f'{path.name},moment_{moment+1:02d},0')
    (root/'labels.csv').write_text('\n'.join(labels)+'\n', encoding='utf-8')
    files = list(root.rglob('*.png'))
    unique = {hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    print(f'{len(files)} images, {len(unique)} unique SHA-256, {len(files)-len(unique)} exact duplicate extras')

if __name__ == '__main__':
    main()
