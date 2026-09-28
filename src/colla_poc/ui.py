from __future__ import annotations
from pathlib import Path
import json
import pandas as pd
import gradio as gr
import numpy as np

def build_ui(run: Path) -> gr.Blocks:
    photos = pd.read_parquet(run/'photos.parquet').fillna('')
    records = photos.to_dict('records')
    by_id = {p['photo_id']: p for p in records}
    moments = json.loads((run/'moments.json').read_text())
    dup = json.loads((run/'duplicates.json').read_text())
    recs = {r['moment_id']: r for r in json.loads((run/'recommendations.json').read_text())}
    metrics = json.loads((run/'metrics.json').read_text())
    details = json.loads((run/'photo_details.json').read_text())
    def gallery(ids: list[str], recommended: set[str] | None = None):
        recommended = recommended or set()
        return [(str(run/by_id[i]['thumbnail']), f"{'★ Recommended · ' if i in recommended else ''}{by_id[i]['filename']} · 품질 {by_id[i]['quality_score']:.2f} · {by_id[i]['semantic_category']} · {by_id[i]['taken_at']}") for i in ids]
    def show_moment(mid: str):
        m = next(m for m in moments if m['moment_id'] == mid)
        return f"{m['start_time']} ~ {m['end_time']} · {m['photo_count']}장 · GPS {m['latitude']}, {m['longitude']}", gallery(m['photo_ids'], set(recs[mid]['top_5']))
    def show_similar(group_id: str):
        g = next(g for g in dup['similar'] if g['group_id'] == group_id)
        winner = max(g['photo_ids'], key=lambda pid: by_id[pid]['quality_score'])
        return gallery(g['photo_ids'], {winner})
    def debug(pid: str):
        p = by_id[pid]; vector = np.load(run/p['embedding_cache_path'])
        candidates = []
        for q in records:
            if q['photo_id'] == pid:
                continue
            other = np.load(run/q['embedding_cache_path'])
            candidates.append((q['filename'], float(vector @ other)))
        candidates.sort(key=lambda x: x[1], reverse=True)
        return json.dumps({'photo': p, 'exif': details.get(pid), 'nearest': candidates[:5]}, indent=2, ensure_ascii=False, default=str)
    with gr.Blocks(title='Colla Image Curation PoC') as demo:
        gr.Markdown('# Colla Image Curation PoC')
        with gr.Tab('Overview'):
            gr.Markdown(f"사진 **{metrics['photo_count']}**장 · Moment **{metrics['moment_count']}**개 · 완전 중복 **{metrics['exact_duplicate_groups']}**그룹 · 유사 사진 **{metrics['similar_group_count']}**그룹 · 대표 추천 **{metrics['recommendation_count']}**장 · 총 **{metrics['timings']['total']:.1f}초**")
            gr.JSON(metrics['semantic_categories'], label='Semantic categories')
        with gr.Tab('Moments'):
            choice = gr.Dropdown([m['moment_id'] for m in moments], value=moments[0]['moment_id'] if moments else None, label='Moment')
            info = gr.Markdown()
            images = gr.Gallery(label='모든 사진', columns=4)
            choice.change(show_moment, choice, [info, images])
            if moments:
                demo.load(lambda: show_moment(moments[0]['moment_id']), outputs=[info, images])
        with gr.Tab('Similar Photos'):
            groups = dup['similar']
            similar_choice = gr.Dropdown([g['group_id'] for g in groups], value=groups[0]['group_id'] if groups else None, label='비슷한 사진 그룹')
            similar_gallery = gr.Gallery(columns=4)
            similar_choice.change(show_similar, similar_choice, similar_gallery)
            if groups:
                demo.load(lambda: show_similar(groups[0]['group_id']), outputs=similar_gallery)
        with gr.Tab('Best Photos'):
            gr.Markdown('품질과 시각적 다양성을 함께 반영한 Top 3')
            gr.Gallery(value=gallery([pid for r in recs.values() for pid in r['top_3']], {r['representative'] for r in recs.values()}), columns=4)
        with gr.Tab('Debug'):
            photo_choice = gr.Dropdown([p['photo_id'] for p in records], value=records[0]['photo_id'] if records else None, label='Photo')
            debug_data = gr.Code(label='Metadata and nearest photos', language='json')
            photo_choice.change(debug, photo_choice, debug_data)
            if records:
                demo.load(lambda: debug(records[0]['photo_id']), outputs=debug_data)
    return demo
