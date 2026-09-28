from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import json
import pandas as pd
from sklearn.metrics import adjusted_rand_score

def evaluate(run: Path, labels_path: Path) -> dict:
    photos = pd.read_parquet(run/'photos.parquet')
    labels = pd.read_csv(labels_path, dtype={'filename': str, 'moment_id': str})
    if labels['filename'].duplicated().any() or photos['filename'].duplicated().any():
        raise ValueError('Filenames must be unique in both labels and photos for evaluation')
    if 'moment_id' not in labels:
        raise ValueError('labels.csv needs moment_id')
    joined = photos.merge(labels, on='filename', suffixes=('_pred', '_true'))
    if joined.empty:
        raise ValueError('No matching filenames in labels.csv')
    true = list(joined['moment_id_true']); pred = list(joined['moment_id_pred'])
    tp = fp = fn = 0
    for i in range(len(joined)):
        for j in range(i+1, len(joined)):
            same_t, same_p = true[i] == true[j], pred[i] == pred[j]
            tp += bool(same_t and same_p); fp += bool(not same_t and same_p); fn += bool(same_t and not same_p)
    precision = tp/(tp+fp) if tp+fp else 0
    recall = tp/(tp+fn) if tp+fn else 0
    result = dict(labeled_photos=len(joined), adjusted_rand_index=float(adjusted_rand_score(true, pred)),
                  pairwise_precision=precision, pairwise_recall=recall,
                  pairwise_f1=2*precision*recall/(precision+recall) if precision+recall else 0)
    if 'keep' in joined:
        recommendations = {r['moment_id']: r for r in json.loads((run/'recommendations.json').read_text())}
        by_moment = defaultdict(list)
        for _, row in joined.iterrows():
            by_moment[row['moment_id_pred']].append(row)
        hits1 = hits3 = count = 0
        for rows in by_moment.values():
            keep = {r['photo_id'] for r in rows if str(r['keep']).lower() in ('1', 'true', 'yes')}
            if not keep:
                continue
            chosen = recommendations[rows[0]['moment_id_pred']]
            count += 1; hits1 += chosen['representative'] in keep
            hits3 += bool(keep.intersection(chosen['top_3']))
        result['top_1_hit_rate'] = hits1/count if count else None
        result['top_3_hit_rate'] = hits3/count if count else None
        result['evaluated_moments'] = count
    if 'semantic_category_true' in joined:
        result['semantic_accuracy'] = float((joined['semantic_category'] == joined['semantic_category_true']).mean())
    return result
