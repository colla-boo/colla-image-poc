from __future__ import annotations
from collections import Counter
import numpy as np

def rank(photos: list[dict], vectors: np.ndarray, indices: list[int], config: dict, prior_counts: Counter[str] | None = None) -> list[int]:
    remaining = set(indices)
    chosen: list[int] = []
    counts: Counter[str] = Counter(prior_counts or {})
    while remaining and len(chosen) < config['top_k']:
        def utility(i: int) -> tuple[float, int]:
            p = photos[i]
            similarity = max((float(vectors[i] @ vectors[j]) for j in chosen), default=0)
            balance = 0.0
            if chosen and p.get('photographer_id'):
                balance = config.get('contributor_balance_weight', 0) / (1 + counts[p['photographer_id']])
            score = config['quality_weight'] * p['quality_score'] - (config['similarity_penalty'] * similarity if chosen else 0) + balance
            return score, -i
        winner = max(remaining, key=utility)
        remaining.remove(winner)
        chosen.append(winner)
        if photos[winner].get('photographer_id'):
            counts[photos[winner]['photographer_id']] += 1
    return chosen

def recommendations(photos: list[dict], vectors: np.ndarray, moments: list[dict], config: dict) -> list[dict]:
    by_id = {p['photo_id']: i for i, p in enumerate(photos)}
    result = []
    representative_counts: Counter[str] = Counter()
    for moment in moments:
        indices = [by_id[p] for p in moment['photo_ids']]
        ranked = rank(photos, vectors, indices, config, representative_counts)
        photographer = photos[ranked[0]].get('photographer_id')
        if photographer:
            representative_counts[photographer] += 1
        result.append(dict(moment_id=moment['moment_id'], representative=photos[ranked[0]]['photo_id'],
                           top_3=[photos[i]['photo_id'] for i in ranked[:3]],
                           top_5=[photos[i]['photo_id'] for i in ranked[:5]]))
    return result
