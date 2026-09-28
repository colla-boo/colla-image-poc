"""Render one source-backed 16:9 summary image from a Colla analysis run."""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import argparse
import json
import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch

FONT_PATH = Path('/System/Library/Fonts/Supplemental/AppleGothic.ttf')
if FONT_PATH.exists():
    font_manager.fontManager.addfont(str(FONT_PATH))
    plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(FONT_PATH)).get_name()
plt.rcParams['axes.unicode_minus'] = False


def pair_similarity(ids: list[str], photos: dict[str, dict], run: Path) -> float:
    vectors = [np.load(run / photos[pid]['embedding_cache_path']) for pid in ids]
    return float(np.mean([vectors[i] @ vectors[j] for i in range(len(vectors)) for j in range(i+1, len(vectors))]))


def make_image(run: Path, output: Path) -> None:
    metrics = json.loads((run/'metrics.json').read_text())
    moments = json.loads((run/'moments.json').read_text())
    recommendations = json.loads((run/'recommendations.json').read_text())
    photos = {p['photo_id']: p for p in pd.read_parquet(run/'photos.parquet').to_dict('records')}
    if len(moments) != 8:
        raise ValueError('This 16:9 layout expects exactly eight Moments')
    sha_groups = defaultdict(list)
    for photo in photos.values():
        sha_groups[photo['sha256']].append(photo)
    baseline, diverse = [], []
    for moment, rec in zip(moments, recommendations, strict=True):
        quality_top3 = sorted(moment['photo_ids'], key=lambda pid: photos[pid]['quality_score'], reverse=True)[:3]
        baseline.append(pair_similarity(quality_top3, photos, run))
        diverse.append(pair_similarity(rec['top_3'], photos, run))
    old_similarity, new_similarity = float(np.mean(baseline)), float(np.mean(diverse))

    ink, muted, blue, background = '#11243d', '#607087', '#356ee8', '#f4f7fc'
    fig = plt.figure(figsize=(16, 9), dpi=120, facecolor=background)
    fig.text(.055, .934, 'Colla', fontsize=30, fontweight='bold', color=blue, va='center')
    fig.text(.155, .934, '사진 자동 큐레이션 PoC', fontsize=29, fontweight='bold', color=ink, va='center')
    fig.text(.057, .871, '4명의 사진 400장  →  8개 Moment  →  Moment별 대표 사진 1장', fontsize=17, color=muted)
    fig.text(.948, .94, '합성 데이터 · 로컬 AI 추론', fontsize=12, color=blue, ha='right', va='center')

    cards = [
        (f"{metrics['photo_count']}장", '분석한 사진'),
        (f"{len(sha_groups)}장", '서로 다른 이미지'),
        (f"{metrics['exact_duplicate_extra_photos']}장", '완전 중복 탐지'),
        (f"{metrics['moment_count']}개", '자동 생성한 Moment'),
    ]
    for i, (number, label) in enumerate(cards):
        x = .055 + i*.231
        box = FancyBboxPatch((x, .735), .213, .108, boxstyle='round,pad=0.006,rounding_size=0.018',
                             transform=fig.transFigure, facecolor='white', edgecolor='#e3e9f3', linewidth=1.2)
        fig.add_artist(box)
        fig.text(x+.017, .797, number, fontsize=25, fontweight='bold', color=ink, va='center')
        fig.text(x+.017, .755, label, fontsize=13, color=muted, va='center')
    fig.text(.057, .688, 'Moment별 대표 사진', fontsize=19, fontweight='bold', color=ink)
    fig.text(.948, .688, '각 Moment 50장 중 추천 1장', fontsize=12, color=muted, ha='right')

    for index, (moment, recommendation) in enumerate(zip(moments, recommendations, strict=True)):
        row, col = divmod(index, 4)
        x, y = .055 + col*.231, .409 - row*.260
        tile = FancyBboxPatch((x, y), .213, .242, boxstyle='round,pad=0.006,rounding_size=0.014',
                              transform=fig.transFigure, facecolor='white', edgecolor='#e3e9f3', linewidth=1)
        fig.add_artist(tile)
        photo = photos[recommendation['representative']]
        image = Image.open(run / photo['thumbnail'])
        ax = fig.add_axes([x+.012, y+.067, .189, .16], zorder=2)
        ax.imshow(image)
        ax.set_axis_off()
        fig.text(x+.013, y+.034, f"Moment {index+1:02d}", fontsize=14, fontweight='bold', color=ink, va='center')
        fig.text(x+.200, y+.034, f"{moment['photo_count']}장", fontsize=12, color=muted, ha='right', va='center')

    fig.text(.057, .082, f'대표 후보 Top 3의 평균 사진 간 유사도  {old_similarity:.3f} → {new_similarity:.3f}',
             fontsize=17, fontweight='bold', color=blue)
    fig.text(.057, .047, '※ 재현용 합성 도형 사진입니다. 실제 여행·모임 사진에서의 정확도는 별도 검증이 필요합니다.', fontsize=12, color=muted)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=120, facecolor=background)
    plt.close(fig)
    print(output)
    print(f'{old_similarity:.3f} -> {new_similarity:.3f}')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    output = args.output or run/'ppt-summary.png'
    make_image(run, output.resolve())

if __name__ == '__main__':
    main()
