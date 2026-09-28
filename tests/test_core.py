from datetime import datetime
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from colla_poc.ingest import sha256, parse_exif_datetime, metadata
from colla_poc.vision import phash, phash_distance, quality
from colla_poc.clustering import haversine, moment_similarity, duplicates, moments
from colla_poc.ranking import rank

def test_hash_and_phash(tmp_path: Path):
    image = Image.new('RGB', (128,128), 'white')
    ImageDraw.Draw(image).rectangle((20,20,60,60), fill='blue')
    a,b = tmp_path/'a.png',tmp_path/'b.png'
    image.save(a); b.write_bytes(a.read_bytes())
    assert sha256(a) == sha256(b)
    assert phash_distance(phash(image), phash(image.copy())) == 0

def test_exif_and_fallback(tmp_path: Path):
    assert parse_exif_datetime('2026:09:28 10:20:30') == datetime(2026,9,28,10,20,30)
    assert parse_exif_datetime('bad') is None
    p = tmp_path/'plain.png'; Image.new('RGB',(32,32)).save(p)
    assert metadata(p)['taken_at_source'] == 'filesystem_mtime'
    assert metadata(p)['latitude'] is None

def test_quality_components():
    w = dict(sharpness_weight=.45, exposure_weight=.25, contrast_weight=.15, resolution_weight=.15)
    scores = quality(Image.new('RGB',(1200,1200),'gray'), w)
    assert all(0 <= v <= 1 for v in scores.values())
    assert abs(scores['quality_score'] - sum(scores[f'{k}_score']*w[f'{k}_weight'] for k in ('sharpness','exposure','contrast','resolution'))) < 1e-9

def test_haversine_and_missing_gps():
    assert haversine(0,0,0,0) == 0
    a = {'taken_at':'2026-09-28T12:00:00','latitude':None,'longitude':None}
    b = {'taken_at':'2026-09-28T12:01:00','latitude':None,'longitude':None}
    config = dict(time_weight=.5, visual_weight=.35, gps_weight=.15,max_time_distance_minutes=60,max_gps_distance_meters=2000)
    assert moment_similarity(a,b,np.array([1.,0]),np.array([1.,0]),config) > .95

def test_ranking_diversity():
    photos = [dict(quality_score=q, photographer_id=None) for q in (1.,.95,.80)]
    vectors = np.array([[1.,0.],[1.,0.],[0.,1.]])
    ids = rank(photos,vectors,[0,1,2],dict(top_k=3,quality_weight=1.,similarity_penalty=.35))
    assert ids == [0,2,1]

def test_groups_and_moments():
    photos = [dict(photo_id=f'p{i}',sha256='same' if i<2 else 'other',phash='0000000000000000' if i<2 else 'ffffffffffffffff',taken_at=f'2026-09-28T{hour:02d}:00:00',latitude=None,longitude=None,quality_score=.8) for i,hour in enumerate((10,10,12))]
    vec = np.array([[1.,0.],[1.,0.],[0.,1.]])
    dup = duplicates(photos,vec,dict(candidate_time_window_minutes=180,phash_max_distance=8,cosine_similarity_threshold=.94))
    assert len(dup['exact']) == 1
    assert len(dup['similar']) == 1
    config = dict(coarse_time_gap_minutes=45,time_weight=.5,visual_weight=.35,gps_weight=.15,max_time_distance_minutes=60,max_gps_distance_meters=2000,split_similarity_threshold=.49)
    assert len(moments(photos,vec,config)) == 2
