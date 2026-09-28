"""Create a tiny, labeled photo fixture without external assets."""
from __future__ import annotations
from datetime import datetime
from pathlib import Path
import argparse
import os
from PIL import Image, ImageDraw

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    parser.add_argument('--copies', type=int, default=0, help='Extra byte-identical copies for scale checks')
    args = parser.parse_args()
    root = args.output.resolve()
    for person in ('amy', 'ben'):
        (root/person).mkdir(parents=True, exist_ok=True)
    originals = []
    for i in range(12):
        group = i // 6
        image = Image.new('RGB', (256, 192), '#90c6e8' if group == 0 else '#d7a66a')
        draw = ImageDraw.Draw(image)
        if group == 0:
            draw.ellipse((55+i%3*2, 30, 170, 140), fill='#195b8e')
        else:
            draw.rectangle((55+i%3*2, 30, 170, 140), fill='#6b3520')
        path = root/('amy' if i%2 == 0 else 'ben')/f'image_{i:02d}.png'
        image.save(path)
        stamp = datetime(2026, 9, 28, 10 if group == 0 else 12, i%6).timestamp()
        os.utime(path, (stamp, stamp))
        originals.append(path)
    duplicate = root/'ben'/'copy_00.png'
    duplicate.write_bytes(originals[0].read_bytes())
    stamp = datetime(2026, 9, 28, 10, 1).timestamp()
    os.utime(duplicate, (stamp, stamp))
    with (root/'labels.csv').open('w', encoding='utf-8') as stream:
        stream.write('filename,moment_id,keep\n')
        for i in range(12):
            stream.write(f'image_{i:02d}.png,{"first" if i < 6 else "second"},{int(i%6 == 0)}\n')
        stream.write('copy_00.png,first,0\n')
    extra = root/'scale'
    if args.copies:
        extra.mkdir(exist_ok=True)
        for i in range(args.copies):
            source = originals[i%12]
            target = extra/f'copy_{i:04d}.png'
            target.write_bytes(source.read_bytes())
            stamp = datetime(2026, 9, 28, 10 if i%12 < 6 else 12, i%6).timestamp()
            os.utime(target, (stamp, stamp))
    print(f'Created {13 + args.copies} images in {root}')

if __name__ == '__main__':
    main()
