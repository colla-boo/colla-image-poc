from __future__ import annotations
from datetime import datetime
from pathlib import Path
import hashlib
from PIL import Image, ImageOps
import pillow_heif

pillow_heif.register_heif_opener()
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.heic', '.heif'}

def scan(root: Path, output: Path) -> list[Path]:
    root, output = root.resolve(), output.resolve()
    if not root.is_dir():
        raise ValueError(f'Input folder does not exist: {root}')
    return sorted(p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in EXTENSIONS and not p.resolve().is_relative_to(output))

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def parse_exif_datetime(value: object) -> datetime | None:
    if not value:
        return None
    if isinstance(value, bytes):
        value = value.decode(errors='replace')
    for fmt in ('%Y:%m:%d %H:%M:%S', '%Y-%m-%d %H:%M:%S'):
        try:
            return datetime.strptime(str(value).strip(), fmt)
        except ValueError:
            pass
    return None

def _decimal_gps(value: object, ref: object) -> float | None:
    try:
        parts = list(value)
        result = sum(float(parts[i]) / (60 ** i) for i in range(3))
        return -result if str(ref).upper() in ('S', 'W') else result
    except (ValueError, TypeError, ZeroDivisionError):
        return None

def metadata(path: Path) -> dict:
    stat = path.stat()
    with Image.open(path) as source:
        exif = source.getexif()
        gps = exif.get_ifd(34853) if 34853 in exif else {}
        dt = parse_exif_datetime(exif.get(36867))
        origin = 'exif_datetime_original'
        if dt is None:
            dt = parse_exif_datetime(exif.get(36868) or exif.get(306))
            origin = 'exif_create_date'
        if dt is None:
            dt = datetime.fromtimestamp(stat.st_mtime)
            origin = 'filesystem_mtime'
        return dict(path=str(path.resolve()), filename=path.name, width=source.width, height=source.height,
                    file_size=stat.st_size, taken_at=dt.isoformat(), taken_at_source=origin,
                    latitude=_decimal_gps(gps.get(2), gps.get(1)) if gps else None,
                    longitude=_decimal_gps(gps.get(4), gps.get(3)) if gps else None,
                    camera_maker=str(exif.get(271, '')), camera_model=str(exif.get(272, '')),
                    orientation=exif.get(274), exif={str(k): str(v) for k, v in exif.items() if k != 34853})

def open_rgb(path: Path) -> Image.Image:
    with Image.open(path) as source:
        return ImageOps.exif_transpose(source).convert('RGB')
