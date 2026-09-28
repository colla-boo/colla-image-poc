from __future__ import annotations
from collections import Counter
from pathlib import Path
from time import perf_counter
import hashlib
import json
import logging
from datetime import datetime
import numpy as np
import pandas as pd
import yaml
from .config import load_config
from .ingest import scan, sha256, metadata, open_rgb
from .vision import phash, quality, thumbnail, SiglipEncoder
from .clustering import duplicates, moments
from .ranking import recommendations
from .reporting import write_report

log = logging.getLogger(__name__)

def _key(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:16]

def _write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding='utf-8')

def analyze(input_dir: Path, output: Path, config_path: Path | None = None, source_directory_as_user: bool = False) -> dict:
    config = load_config(config_path)
    root = input_dir.resolve(); output = output.resolve()
    if root == output:
        raise ValueError('Output must differ from input')
    output.mkdir(parents=True, exist_ok=True)
    (output/'embeddings').mkdir(exist_ok=True)
    (output/'thumbnails').mkdir(exist_ok=True)
    (output/'cache').mkdir(exist_ok=True)
    (output/'config.yaml').write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding='utf-8')
    timings = {}; errors = []; start = perf_counter()
    stage = perf_counter()
    paths = scan(root, output)
    timings['scan'] = perf_counter()-stage
    log.info('[1/8] 사진 탐색: %d개', len(paths))
    stage = perf_counter()
    photos = []
    for path in paths:
        try:
            digest = sha256(path)
            meta_cache = output/'cache'/f'{digest}_metadata.json'
            if meta_cache.exists():
                item = json.loads(meta_cache.read_text(encoding='utf-8'))
                item.update(path=str(path.resolve()), filename=path.name, file_size=path.stat().st_size)
                if item['taken_at_source'] == 'filesystem_mtime':
                    item['taken_at'] = datetime.fromtimestamp(path.stat().st_mtime).isoformat()
            else:
                item = metadata(path)
                meta_cache.write_text(json.dumps(item, ensure_ascii=False), encoding='utf-8')
            item['sha256'] = digest
            item['photo_id'] = f'photo_{len(photos)+1:05d}'
            item['photographer_id'] = path.relative_to(root).parts[0] if source_directory_as_user and len(path.relative_to(root).parts) > 1 else None
            # Decode once so damaged files are excluded before inference.
            open_rgb(path).load()
            photos.append(item)
        except Exception as exc:
            errors.append({'path': str(path), 'stage': 'metadata', 'error': str(exc)})
            log.warning('Skipping %s: %s', path, exc)
    timings['metadata'] = perf_counter()-stage
    log.info('[2/8] Metadata 분석: %d개 성공', len(photos))
    stage = perf_counter()
    visual_key = _key({'algorithm_version': 2, 'quality': config['quality']})
    valid = []
    for p in photos:
        path = Path(p['path']); cache = output/'cache'/f"{p['sha256']}_{visual_key}.json"
        try:
            if cache.exists():
                values = json.loads(cache.read_text(encoding='utf-8'))
            else:
                image = open_rgb(path)
                values = {'phash': phash(image), **quality(image.copy(), config['quality'])}
                cache.write_text(json.dumps(values), encoding='utf-8')
            p.update(values)
            thumb = output/'thumbnails'/f"{p['sha256']}.webp"
            if not thumb.exists():
                thumbnail(open_rgb(path), thumb)
            p['thumbnail'] = str(thumb.relative_to(output))
            valid.append(p)
        except Exception as exc:
            errors.append({'path': str(path), 'stage': 'visual', 'error': str(exc)})
            log.warning('Skipping %s: %s', path, exc)
    photos = valid
    timings['hash_quality_thumbnail'] = perf_counter()-stage
    log.info('[3/8] Hash 계산 / [4/8] 품질 분석: %.2fs', timings['hash_quality_thumbnail'])
    if not photos:
        _write_json(output/'errors.json', errors)
        raise ValueError('No readable photos found')
    stage = perf_counter()
    emb_config = config['embedding']; labels = config['semantics']['labels']
    model_key = _key({'model': emb_config['model'], 'revision': emb_config['revision']})
    semantic_key = _key({'model': model_key, 'labels': labels})
    vectors: list[np.ndarray | None] = [None] * len(photos)
    categories: list[tuple[str, float] | None] = [None] * len(photos)
    missing = []
    missing_by_sha: dict[str, list[int]] = {}
    for i, p in enumerate(photos):
        vector_file = output/'embeddings'/f"{p['sha256']}_{model_key}.npy"
        semantic_file = output/'cache'/f"{p['sha256']}_{semantic_key}_semantic.json"
        p['embedding_cache_path'] = str(vector_file.relative_to(output))
        if vector_file.exists():
            vectors[i] = np.load(vector_file)
        if semantic_file.exists():
            obj = json.loads(semantic_file.read_text())
            categories[i] = (obj['category'], obj['confidence'])
        if vectors[i] is None or categories[i] is None:
            if p['sha256'] not in missing_by_sha:
                missing.append(i)
                missing_by_sha[p['sha256']] = []
            missing_by_sha[p['sha256']].append(i)
    if missing:
        encoder = SiglipEncoder(emb_config['model'], emb_config['revision'], labels)
        for offset in range(0, len(missing), emb_config['batch_size']):
            batch = missing[offset:offset+emb_config['batch_size']]
            try:
                batch_vectors, batch_categories = encoder.encode([Path(photos[i]['path']) for i in batch])
            except Exception as exc:
                if len(batch) == 1:
                    errors.append({'path': photos[batch[0]]['path'], 'stage': 'embedding', 'error': str(exc)})
                    continue
                # Isolate individual bad images without losing the rest of the batch.
                for i in batch:
                    try:
                        v, c = encoder.encode([Path(photos[i]['path'])])
                        vectors[i], categories[i] = v[0], c[0]
                    except Exception as inner:
                        errors.append({'path': photos[i]['path'], 'stage': 'embedding', 'error': str(inner)})
                continue
            for position, i in enumerate(batch):
                vectors[i], categories[i] = batch_vectors[position], batch_categories[position]
        for representative in missing:
            for duplicate_index in missing_by_sha[photos[representative]['sha256']]:
                vectors[duplicate_index] = vectors[representative]
                categories[duplicate_index] = categories[representative]
    good = []; final_vectors = []
    for i, p in enumerate(photos):
        if vectors[i] is None or categories[i] is None:
            continue
        vec = np.asarray(vectors[i], dtype=np.float32)
        vec /= max(float(np.linalg.norm(vec)), 1e-12)
        vector_file = output/p['embedding_cache_path']
        if not vector_file.exists():
            np.save(vector_file, vec)
        semantic_file = output/'cache'/f"{p['sha256']}_{semantic_key}_semantic.json"
        if not semantic_file.exists():
            semantic_file.write_text(json.dumps({'category': categories[i][0], 'confidence': categories[i][1], 'model_id': emb_config['model'], 'model_revision': emb_config['revision']}))
        category, confidence = categories[i]
        p['semantic_category'] = category if confidence >= config['semantics']['confidence_threshold'] else 'other'
        p['semantic_confidence'] = confidence
        good.append(p); final_vectors.append(vec)
    photos = good; matrix = np.stack(final_vectors) if final_vectors else np.zeros((0, 1), dtype=np.float32)
    timings['embedding_semantics'] = perf_counter()-stage
    log.info('[5/8] Image embedding 생성: %d개', len(photos))
    if not photos:
        _write_json(output/'errors.json', errors)
        raise ValueError('No photos could be embedded')
    stage = perf_counter()
    dup = duplicates(photos, matrix, config['duplicate'])
    timings['duplicates'] = perf_counter()-stage
    log.info('[6/8] 중복/유사 사진 분석: %d / %d 그룹', len(dup['exact']), len(dup['similar']))
    stage = perf_counter()
    moment_list = moments(photos, matrix, config['moment'])
    timings['moments'] = perf_counter()-stage
    log.info('[7/8] Moment clustering: %d개', len(moment_list))
    stage = perf_counter()
    recs = recommendations(photos, matrix, moment_list, config['ranking'])
    timings['ranking'] = perf_counter()-stage
    log.info('[8/8] 대표 사진 추천: %d개', len(recs))
    timings['total'] = perf_counter()-start
    dataframe = pd.DataFrame([{k:v for k,v in p.items() if k != 'exif'} for p in photos])
    dataframe.to_parquet(output/'photos.parquet', index=False)
    _write_json(output/'photo_details.json', {p['photo_id']: p['exif'] for p in photos})
    _write_json(output/'moments.json', moment_list)
    _write_json(output/'duplicates.json', dup)
    _write_json(output/'recommendations.json', recs)
    _write_json(output/'errors.json', errors)
    metrics = dict(photo_count=len(photos), moment_count=len(moment_list), exact_duplicate_groups=len(dup['exact']),
                   exact_duplicate_extra_photos=sum(len(g['photo_ids'])-1 for g in dup['exact']),
                   similar_group_count=len(dup['similar']), recommendation_count=len(recs),
                   gps_coverage=sum(p['latitude'] is not None and p['longitude'] is not None for p in photos)/len(photos),
                   semantic_categories=dict(Counter(p['semantic_category'] for p in photos)),
                   photographer_counts=dict(Counter(p['photographer_id'] for p in photos if p['photographer_id'])),
                   recommended_photographer_counts=dict(Counter(next(p['photographer_id'] for p in photos if p['photo_id']==r['representative']) for r in recs if next(p['photographer_id'] for p in photos if p['photo_id']==r['representative']))),
                   timings=timings, error_count=len(errors), model_id=emb_config['model'], model_revision=emb_config['revision'])
    _write_json(output/'metrics.json', metrics)
    write_report(output, photos, moment_list, dup, recs, metrics, config)
    return metrics
