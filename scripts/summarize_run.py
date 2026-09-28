"""Inspect ranking diversity and exact duplicate recovery for an existing run."""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd

def mean_pair_similarity(ids: list[str], by_id: dict[str, dict], run: Path) -> float:
    vectors = [np.load(run / by_id[pid]['embedding_cache_path']) for pid in ids]
    pairs = [float(vectors[i] @ vectors[j]) for i in range(len(vectors)) for j in range(i+1, len(vectors))]
    return float(np.mean(pairs)) if pairs else 0.0

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    photos = pd.read_parquet(run/'photos.parquet').to_dict('records')
    by_id = {p['photo_id']:p for p in photos}
    moments = json.loads((run/'moments.json').read_text())
    recommendations = {r['moment_id']:r for r in json.loads((run/'recommendations.json').read_text())}
    sha_groups = defaultdict(list)
    for p in photos:
        sha_groups[p['sha256']].append(p)
    expected_exact = [g for g in sha_groups.values() if len(g)>1]
    exact_recovered = sum(len({p['duplicate_group_id'] for p in group}) == 1 and bool(group[0]['duplicate_group_id']) for group in expected_exact)
    baseline_similarity = []; mmr_similarity = []; baseline_quality = []; mmr_quality = []
    for moment in moments:
        ids = moment['photo_ids']
        baseline = sorted(ids, key=lambda pid: by_id[pid]['quality_score'], reverse=True)[:3]
        chosen = recommendations[moment['moment_id']]['top_3']
        baseline_similarity.append(mean_pair_similarity(baseline, by_id, run))
        mmr_similarity.append(mean_pair_similarity(chosen, by_id, run))
        baseline_quality.append(float(np.mean([by_id[pid]['quality_score'] for pid in baseline])))
        mmr_quality.append(float(np.mean([by_id[pid]['quality_score'] for pid in chosen])))
    result = {
        'photos':len(photos), 'unique_sha256':len(sha_groups),
        'exact_duplicate_groups_expected':len(expected_exact),
        'exact_duplicate_groups_recovered':exact_recovered,
        'moment_sizes':[m['photo_count'] for m in moments],
        'quality_only_top3_mean_pair_cosine':float(np.mean(baseline_similarity)),
        'mmr_top3_mean_pair_cosine':float(np.mean(mmr_similarity)),
        'quality_only_top3_mean_quality':float(np.mean(baseline_quality)),
        'mmr_top3_mean_quality':float(np.mean(mmr_quality)),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == '__main__':
    main()
