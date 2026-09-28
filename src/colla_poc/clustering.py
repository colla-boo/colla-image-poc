from __future__ import annotations
from collections import defaultdict
from datetime import datetime
from math import asin, cos, radians, sin, sqrt
import numpy as np
from .vision import phash_distance

class UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))
    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x
    def join(self, a: int, b: int) -> None:
        self.parent[self.find(a)] = self.find(b)

def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    dlat, dlon = radians(lat2-lat1), radians(lon2-lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    return 6371000 * 2 * asin(min(1, sqrt(a)))

def _time(photo: dict) -> datetime | None:
    try:
        return datetime.fromisoformat(photo['taken_at']) if photo.get('taken_at') else None
    except ValueError:
        return None

def _groups(uf: UnionFind, prefix: str, photos: list[dict], field: str) -> list[dict]:
    buckets: dict[int, list[int]] = defaultdict(list)
    for i in range(len(photos)):
        buckets[uf.find(i)].append(i)
    groups = []
    for members in sorted((m for m in buckets.values() if len(m) > 1), key=lambda m: min(m)):
        group_id = f'{prefix}_{len(groups)+1:03d}'
        for index in members:
            photos[index][field] = group_id
        groups.append({'group_id': group_id, 'photo_ids': [photos[i]['photo_id'] for i in members]})
    return groups

def duplicates(photos: list[dict], vectors: np.ndarray, config: dict) -> dict:
    exact = UnionFind(len(photos))
    by_hash: dict[str, int] = {}
    for i, photo in enumerate(photos):
        if photo['sha256'] in by_hash:
            exact.join(i, by_hash[photo['sha256']])
        by_hash[photo['sha256']] = i
    exact_groups = _groups(exact, 'exact', photos, 'duplicate_group_id')
    for photo in photos:
        photo['duplicate_type'] = 'exact' if photo.get('duplicate_group_id') else None
    similar = UnionFind(len(photos))
    # Sort by time and compare only a bounded time window; exact hashes work across all dates.
    ordered = sorted(range(len(photos)), key=lambda i: photos[i].get('taken_at') or '')
    window = config['candidate_time_window_minutes'] * 60
    for pos, i in enumerate(ordered):
        ti = _time(photos[i])
        for j in ordered[pos+1:]:
            tj = _time(photos[j])
            if ti and tj and (tj-ti).total_seconds() > window:
                break
            if photos[i]['sha256'] == photos[j]['sha256']:
                similar.join(i, j)
                continue
            if phash_distance(photos[i]['phash'], photos[j]['phash']) <= config['phash_max_distance'] or float(vectors[i] @ vectors[j]) >= config['cosine_similarity_threshold']:
                similar.join(i, j)
    return {'exact': exact_groups, 'similar': _groups(similar, 'similar', photos, 'similar_group_id')}

def moment_similarity(a: dict, b: dict, va: np.ndarray, vb: np.ndarray, config: dict) -> float:
    values = []
    ta, tb = _time(a), _time(b)
    if ta and tb:
        minutes = abs((ta-tb).total_seconds()) / 60
        values.append((config['time_weight'], max(0., 1 - minutes / config['max_time_distance_minutes'])))
    values.append((config['visual_weight'], max(0., float(va @ vb))))
    if all(p.get('latitude') is not None and p.get('longitude') is not None for p in (a, b)):
        meters = haversine(a['latitude'], a['longitude'], b['latitude'], b['longitude'])
        values.append((config['gps_weight'], max(0., 1 - meters/config['max_gps_distance_meters'])))
    total = sum(w for w, _ in values)
    return sum(w*s for w, s in values) / total if total else 0

def moments(photos: list[dict], vectors: np.ndarray, config: dict) -> list[dict]:
    if not photos:
        return []
    ordered = sorted(range(len(photos)), key=lambda i: photos[i].get('taken_at') or '')
    segments: list[list[int]] = []
    for i in ordered:
        if not segments:
            segments.append([i]); continue
        previous = segments[-1][-1]
        ta, tb = _time(photos[previous]), _time(photos[i])
        gap = (tb-ta).total_seconds()/60 if ta and tb else 0
        score = moment_similarity(photos[previous], photos[i], vectors[previous], vectors[i], config)
        if gap > config['coarse_time_gap_minutes'] or score < config['split_similarity_threshold']:
            segments.append([i])
        else:
            segments[-1].append(i)
    result = []
    for members in segments:
        mid = f'moment_{len(result)+1:03d}'
        selected = [photos[i] for i in members]
        for p in selected:
            p['moment_id'] = mid
        lat = [p['latitude'] for p in selected if p.get('latitude') is not None]
        lon = [p['longitude'] for p in selected if p.get('longitude') is not None]
        times = [p['taken_at'] for p in selected if p.get('taken_at')]
        result.append(dict(moment_id=mid, photo_ids=[p['photo_id'] for p in selected], photo_count=len(selected),
                           start_time=min(times) if times else None, end_time=max(times) if times else None,
                           latitude=sum(lat)/len(lat) if lat else None, longitude=sum(lon)/len(lon) if lon else None))
    return result
