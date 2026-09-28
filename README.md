# Colla Image Curation PoC

여러 사람이 찍은 사진을 로컬에서 분석해 **Moment 묶기, 중복/유사 사진 묶기, 대표 사진 추천**이 가능한지 검증하는 실험용 Python 프로그램입니다. 원본 파일은 읽기만 합니다. 계정, 업로드, 자동 삭제는 없습니다.

## 구조와 흐름

`폴더 탐색 → EXIF/파일 메타데이터 → SHA-256/pHash/품질/썸네일 → SigLIP 배치 임베딩과 zero-shot 의미 분류 → 중복·Moment 묶기 → MMR식 대표 추천 → Parquet/JSON/HTML/Gradio`.

- `ingest.py`: JPEG/PNG/WebP/HEIC/HEIF 탐색, EXIF와 GPS, 촬영시간 fallback
- `vision.py`: pHash, 품질 휴리스틱, SigLIP vision/text encoder
- `clustering.py`: SHA 중복, 유사 그룹, 시간·GPS·시각 신호의 Moment
- `ranking.py`: 품질, 시각적 다양성, 선택적 촬영자 균형
- `pipeline.py`: 캐시와 실행 단계
- `reporting.py`, `ui.py`, `evaluation.py`: 결과 검토와 정량 평가

## 설치

Python 3.12 이상과 [uv](https://docs.astral.sh/uv/getting-started/installation/)가 필요합니다. macOS에서는 `brew install uv`도 가능합니다.

```bash
uv sync --extra dev
uv run colla-poc --help
uv run pytest -q
```

첫 분석은 SigLIP 모델 가중치를 Hugging Face에서 다운로드합니다. 다운로드 후 추론은 로컬에서 수행합니다. 외부 SaaS API는 사용하지 않습니다. MPS, CUDA, CPU 순으로 자동 선택합니다. 모델은 `config/default.yaml`에서 바꿀 수 있습니다.

## 실행

```bash
uv run colla-poc analyze ./dataset --output ./runs/trial-01 --config ./config/default.yaml
uv run colla-poc analyze ./dataset --output ./runs/trial-02 --source-directory-as-user
uv run colla-poc ui ./runs/trial-01
uv run colla-poc evaluate ./runs/trial-01 --labels ./labels.csv
```

`--source-directory-as-user`는 입력 폴더의 첫 하위 폴더를 촬영자 ID로 봅니다. 대표 사진 순위에 작은 균형 보너스를 주며 강제 할당량은 아닙니다. 원본 사진은 이동·수정·삭제하지 않습니다. 결과는 지정한 `--output` 아래에만 저장됩니다. 출력 폴더가 입력 폴더 안에 있어도 스캔에서 제외됩니다.

## 출력

- `photos.parquet`: 사진별 메타데이터, 점수, 그룹 ID와 캐시 경로
- `moments.json`, `duplicates.json`, `recommendations.json`: 묶기와 추천
- `metrics.json`, `errors.json`: 실행 결과와 읽기 실패 기록
- `photo_details.json`: 디버그용 EXIF
- `embeddings/`, `thumbnails/`, `cache/`: 재분석 캐시
- `config.yaml`: 이번 실행에 사용한 설정
- `report.html`: 대표 썸네일을 포함한 단일 파일 결과 보고서

같은 출력 폴더에서 다시 실행하면 SHA-256 기준 시각 정보와 모델별 임베딩을 재사용합니다. 품질 설정 변경은 품질 캐시를, 모델 ID/리비전 변경은 임베딩 캐시를, semantic prompt 변경은 분류 캐시를 새로 만듭니다. 모델의 `main` 리비전은 시간이 지나면 바뀔 수 있으므로 재현 실험에는 커밋 SHA를 설정하는 것이 좋습니다.

## 알고리즘과 조정

- **SHA-256**: 파일 바이트가 완전히 같은 경우. **pHash**: 크기나 인코딩이 달라도 시각적으로 매우 비슷한 후보를 찾습니다. **SigLIP 임베딩**: 의미적으로 비슷한 사진도 찾습니다.
- **유사 사진 그룹**은 중복/연사를 나타냅니다. **Moment**는 함께 있었던 시간과 장소의 사진을 묶는 단위입니다. 유사 사진 하나의 Moment에 여러 그룹이 있을 수 있습니다.
- 유사 비교는 촬영 시간 window 내에서만 수행합니다. SHA 중복은 전체 입력에서 탐지합니다. 시간·GPS 없는 경우에는 이용 가능한 신호의 가중치를 다시 정규화합니다.
- Moment는 시간 순서로 정렬한 뒤 큰 시간 간격에서 나누고, 이웃 사진 사이 시간·Haversine GPS 거리·코사인 유사도를 결합해 추가 분할합니다. 설정에서 window와 가중치, 분할 기준을 바꿉니다.
- 품질은 Laplacian 분산, 밝기/클리핑, 명암 대비, 해상도를 0–1로 정규화한 **대표 선정용 휴리스틱**입니다. 절대적인 좋은 사진 판정이 아닙니다.
- 첫 대표는 품질 최고점, 이후에는 `quality_weight × quality − similarity_penalty × 이미 선택된 사진과 최대 유사도 + 촬영자 균형 보너스`로 Top 5를 고릅니다.
- semantic label은 SigLIP 이미지와 text prompt 유사도의 softmax입니다. 표준적인 확률 보정값은 아닙니다.

## 사람 평가

`labels.csv` 예:

```csv
filename,moment_id,keep,semantic_category_true
IMG_001.jpg,beach,1,landscape
IMG_002.jpg,beach,0,people
IMG_003.jpg,cafe,1,food
```

파일명은 유일해야 합니다. `evaluate`는 조정 Rand 지수, 쌍별 precision/recall/F1, Top-1/Top-3 hit rate를 계산합니다. 실제 촬영자별 데이터에 사람이 Moment와 keep을 지정해 평가해야 Colla 적용 가능성을 판단할 수 있습니다.

## 한계

- 촬영시간 fallback으로 mtime을 쓰면 복사 시 시각이 달라질 수 있습니다. 시간대가 없는 EXIF는 지역 시간을 그대로 취급합니다.
- Moment는 설명 가능한 인접 사진 기준 휴리스틱입니다. 긴 행사, 이동 중 촬영, 교차 업로드 시간대에서는 과도하게 나뉘거나 합쳐질 수 있습니다.
- 시간 window 밖의 시각적 유사 사진은 유사 그룹에서 빠질 수 있습니다. 약 2,000장까지 실험을 목표로 하며 극단적으로 짧은 시간에 몰린 사진은 후보 쌍 수가 커집니다.
- SigLIP zero-shot label과 품질 점수는 실제 사용자 선호를 대체하지 않습니다. 평가 데이터 없이 정확도를 주장할 수 없습니다.
- 캐시는 출력 폴더 단위입니다. 다른 출력 폴더 간에는 공유되지 않습니다.

## 재현된 검증 결과

[VALIDATION.md](VALIDATION.md)에 13장, 300장 반복 복사, 400장 중 고유 이미지 360장 검사 수치와 확인하지 못한 범위를 기록했습니다. 재현용 이미지는 `scripts/make_synthetic_fixture.py`와 `scripts/make_scale_fixture.py`로 생성합니다.
