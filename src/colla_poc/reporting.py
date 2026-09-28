from __future__ import annotations
from collections import Counter
from pathlib import Path
import base64
import html
import json

def write_report(output: Path, photos: list[dict], moments: list[dict], duplicates: dict, recommendations: list[dict], metrics: dict, config: dict) -> None:
    by_id = {p['photo_id']: p for p in photos}
    def card(photo_id: str) -> str:
        p = by_id[photo_id]
        data = base64.b64encode((output/p['thumbnail']).read_bytes()).decode()
        return f'<figure><img src="data:image/webp;base64,{data}" alt=""><figcaption>{html.escape(p["filename"])}<br>품질 {p["quality_score"]:.2f} · {html.escape(p["semantic_category"])}</figcaption></figure>'
    blocks = []
    by_moment = {r['moment_id']: r for r in recommendations}
    for m in moments:
        rec = by_moment[m['moment_id']]
        blocks.append(f'<section><h2>{html.escape(m["moment_id"])} · {m["photo_count"]}장</h2><p>{html.escape(str(m["start_time"]))} ~ {html.escape(str(m["end_time"]))} · GPS {m["latitude"]}, {m["longitude"]}</p><div class="grid">'+''.join(card(pid) for pid in rec['top_5'])+'</div></section>')
    quality_buckets = Counter(min(9, int(p['quality_score']*10)) for p in photos)
    bars = ''.join(f'<div><span>{i/10:.1f}–{(i+1)/10:.1f}</span><b style="width:{quality_buckets[i]/max(1,len(photos))*100:.2f}%"></b>{quality_buckets[i]}</div>' for i in range(10))
    categories = ''.join(f'<li>{html.escape(k)}: {v}</li>' for k,v in sorted(metrics['semantic_categories'].items()))
    source = ''.join(f'<li>{html.escape(k)}: 전체 {v}장 / 대표 {metrics["recommended_photographer_counts"].get(k,0)}장</li>' for k,v in metrics['photographer_counts'].items())
    date_range = f"{min(p['taken_at'] for p in photos)} ~ {max(p['taken_at'] for p in photos)}"
    page = f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Colla Image Curation PoC</title><style>body{{font:16px system-ui;margin:0 auto;max-width:1100px;padding:24px;color:#17202a;background:#f6f7f8}}header,section{{background:white;border-radius:14px;padding:22px;margin:16px 0;box-shadow:0 2px 10px #0001}}h1{{margin:0}}.stats,.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px}}.stats div{{background:#e8f1ff;padding:16px;border-radius:10px}}figure{{margin:0}}img{{width:100%;aspect-ratio:1;object-fit:cover;border-radius:8px}}figcaption{{font-size:13px;overflow-wrap:anywhere}}.bars div{{display:flex;gap:8px;align-items:center;margin:4px}}.bars span{{width:65px}}.bars b{{display:block;height:12px;background:#517dd3;min-width:1px}}</style><header><h1>Colla Image Curation PoC</h1><p>로컬 분석 결과 · 품질 수치는 대표 사진 추천용 휴리스틱입니다.</p><div class="stats"><div>사진 <strong>{metrics['photo_count']}</strong>장</div><div>Moment <strong>{metrics['moment_count']}</strong>개</div><div>완전 중복 <strong>{metrics['exact_duplicate_groups']}</strong>그룹</div><div>유사 사진 <strong>{metrics['similar_group_count']}</strong>그룹</div><div>대표 추천 <strong>{metrics['recommendation_count']}</strong>장</div><div>분석 <strong>{metrics['timings']['total']:.1f}</strong>초</div></div><p>촬영 기간 {html.escape(date_range)} · GPS 포함 {metrics['gps_coverage']:.1%} · 오류 {metrics['error_count']}개</p></header><section><h2>분포</h2><h3>품질</h3><div class="bars">{bars}</div><h3>의미 분류</h3><ul>{categories}</ul><h3>촬영자</h3><ul>{source}</ul></section>{''.join(blocks)}<section><h2>처리 시간과 설정</h2><pre>{html.escape(json.dumps(metrics['timings'], indent=2, ensure_ascii=False))}</pre><pre>{html.escape(json.dumps(config, indent=2, ensure_ascii=False))}</pre></section></html>'''
    (output/'report.html').write_text(page, encoding='utf-8')
