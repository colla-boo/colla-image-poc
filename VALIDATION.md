# 검증 기록 — 2026-09-28

## 환경

- macOS Apple Silicon, Python 3.12.11, uv 0.11.12
- SigLIP 모델: `google/siglip-base-patch16-224` (`main`)
- 공개 모델 가중치를 내려받은 뒤 로컬에서 추론. 유료 API 사용 없음.

## 실행한 검사

1. `uv run pytest -q`: **6 passed**
2. `uv run colla-poc --help`: analyze, ui, evaluate 명령 표시
3. `scripts/make_synthetic_fixture.py`로 만든 13장에 대한 전체 분석: **13장, 2 Moment, 완전 중복 6그룹, 유사 그룹 2개, 오류 0**. 첫 실행 155.5초(모델 다운로드 포함); 같은 출력 폴더 재실행 0.03초.
4. `uv run colla-poc evaluate ... --labels ...`: **ARI 1.0, pairwise F1 1.0, Top-1/Top-3 hit rate 1.0** (2개 Moment). 사람이 설계한 단순한 합성 정답이므로 일반 사진 정확도를 뜻하지 않음.
5. Gradio `Blocks` 생성 후 로컬 `127.0.0.1:7879` 서버 응답: **HTTP 200**.
6. 합성 사진 반복 복사를 포함한 **300장** 분석: **13.23초, 2 Moment, 오류 0**. 이 입력은 소수의 고유 이미지와 다수의 SHA 중복으로 구성돼 실제 300장의 서로 다른 사진 추론 속도를 나타내지 않음.
7. 읽을 수 없는 `.jpg` 1개를 섞은 301개 후보: 정상 사진 **300개를 계속 분석**하고 `errors.json`에 실패 **1개** 기록. 손상 파일 제거 후 결과를 재생성함.

## 확대 검사: 400장, 고유 이미지 360장

`scripts/make_scale_fixture.py`로 촬영자 4명, 시간 구간 8개, 구간당 50장(고유 45장과 완전 중복 5장)을 생성했다. 모든 이미지는 외부 자산 없이 Pillow로 만든 합성 도형이다. 이전 300장 검사와 달리 고유 이미지 360장을 실제로 SigLIP으로 추론했다. 이 환경에서 MPS는 사용할 수 없어 CPU로 실행했다.

| 항목 | 결과 |
| --- | ---: |
| 입력/성공/오류 | 400 / 400 / 0 |
| 고유 SHA-256 / 완전 중복 추가 사진 | 360 / 40 |
| 완전 중복 그룹 탐지 | 정답 40 / 탐지 40 |
| 유사 사진 그룹 | 21 |
| Moment | 8개, 각 50장 |
| Moment ARI / pairwise F1 | 1.0 / 1.0 |
| 대표 사진 Top-1 / Top-3 적중률 | 0 / 0.125 |
| 처음 실행 / 같은 출력 폴더 재실행 | 92.02초 / 1.26초 |
| SigLIP 임베딩 단계 / 중복 비교 단계 | 88.26초 / 2.21초 |
| Gradio 로컬 서버 | HTTP 200 |

`keep=1`은 각 구간의 첫 프레임 하나에 임의로 붙인 합성 정답이다. 사람이 실제로 좋은 사진을 고른 정답이 아니므로 Top-1/Top-3 적중률은 대표 추천의 실사용 정확도로 해석할 수 없다. 이 값이 낮은 점은 평가 데이터의 한계를 드러낸다.

Top-3를 품질 점수만으로 고른 경우와 비교하면, 추천 사진끼리의 평균 코사인 유사도는 **0.945 → 0.875**로 낮아졌고 평균 품질 점수는 **0.785 → 0.781**이었다. 이 합성 입력에서는 MMR식 순위가 중복된 구도를 줄이는 방향으로 작동했다. 촬영자는 100장씩 같았으나 Moment 대표 사진은 amy 4장, ben 2장, dana 2장, chris 0장으로 분포했다. 촬영자 균형 보너스가 전역 대표 분포를 보장하지는 않는다.

## 재현

```bash
uv sync --extra dev
uv run python scripts/make_synthetic_fixture.py /tmp/colla-fixture
uv run colla-poc analyze /tmp/colla-fixture --output runs/synthetic --source-directory-as-user
uv run colla-poc evaluate runs/synthetic --labels /tmp/colla-fixture/labels.csv
uv run colla-poc ui runs/synthetic
uv run python scripts/make_synthetic_fixture.py /tmp/colla-fixture-300 --copies 287
uv run colla-poc analyze /tmp/colla-fixture-300 --output runs/scale-300
uv run python scripts/make_scale_fixture.py /tmp/colla-fixture-400
uv run colla-poc analyze /tmp/colla-fixture-400 --output runs/scale-400 --source-directory-as-user
uv run colla-poc evaluate runs/scale-400 --labels /tmp/colla-fixture-400/labels.csv
uv run python scripts/summarize_run.py runs/scale-400
```

## 확인하지 못한 것

- 실제 다중 촬영자 여행/모임 사진 300–1,000장과 사람이 붙인 정답으로 정확도를 평가하지 못함.
- HEIC/HEIF 실사진 디코딩, GPS 포함 실제 EXIF, CUDA 환경은 검증하지 못함.
- 2,000장의 서로 다른 사진이 한 시간대에 몰린 최악의 유사 후보 비교 성능은 검증하지 못함.
