# Dataset Card

## Data Storage Policy

Raw AIHub data is NOT committed to this repository.
Full datasets are stored in GCS: `gs://YOUR_GCS_BUCKET/raw/`

Only schema-compatible synthetic samples and converted subsets are included in this repo under `data/`.

---

## Datasets in Use

### Primary — AIHub 수학 교과 문제 풀이과정 데이터 (No. 30)

- Source: [AIHub dataset 71859](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71859)
- Size: 869 MB
- Records: ~20,319 problem-solving sets
- Language: Korean
- Coverage: Elementary (grades 3-6) + Middle school (grades 1-3) + High school (grade 1)
- Fields: problem text, answer, solution explanation, wrong answer, curriculum standard, difficulty
- GCS location: `gs://YOUR_GCS_BUCKET/raw/aihub_math_30/`
- Converted sample (200 records): `data/aihub/math_problems_sample.jsonl`
- Convert script: `scripts/prepare_aihub_math.py`

### Backup — AIHub 수학 과목 자동 풀이 데이터 (No. 110)

- Source: [AIHub dataset 71716](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71716)
- Size: 15.28 GB
- GCS location: `gs://YOUR_GCS_BUCKET/raw/aihub_math_110/` (to be uploaded from desktop)
- Status: download pending on desktop

### Backup — AIHub 수학 과목 문제생성 데이터 (No. 111)

- Source: [AIHub Education](https://aihub.or.kr)
- Size: 2.88 GB
- GCS location: `gs://YOUR_GCS_BUCKET/raw/aihub_math_111/` (to be uploaded from desktop)
- Status: download pending on desktop

---

## How to Reproduce Locally

1. Request access at [aihub.or.kr](https://aihub.or.kr) (free registration required)
2. Download datasets No. 30, 110, 111 from the Education category
3. Run conversion:

```bash
python scripts/prepare_aihub_math.py \
  --input ~/Downloads/30.수학\ 교과\ 문제\ 풀이과정\ 데이터/3.개방데이터/1.데이터/Training/02.라벨링데이터 \
  --output data/aihub/math_problems.jsonl
```

Or download from GCS (requires `YOUR_GCP_PROJECT` project access):

```bash
gcloud storage cp gs://YOUR_GCS_BUCKET/raw/aihub_math_30/ ./data/raw/ --recursive
```

---

## Synthetic Data (in-repo)

`data/synthetic/` contains schema-compatible samples for testing without AIHub access.
`data/aihub/math_problems_sample.jsonl` contains 200 real converted records.

These are safe to commit. Do not commit raw AIHub zip files.
