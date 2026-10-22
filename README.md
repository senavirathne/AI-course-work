# Fabric defect coursework

Minimal coursework implementation for a textile QA inspector. A trained ANN
classifies a 256 x 256 fabric patch, returns an inspection action, and provides
a Grad-CAM explanation. The study compares the ANN with trivial references, a
tuned HOG/LBP SVM, and SmolVLM-256M on one fixed source-grouped split.

## Install and run

Python 3.11 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[vlm,notebooks]'
FABRIC_CPU_THREADS=1 python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000> and upload a PNG or JPEG from
`data/sample/MVTec/`. Each request runs the included trained model. The score
is an uncalibrated model score, not guaranteed confidence; the UI states the
deployment and out-of-distribution limitations.

## Data

The study uses RAW-FABRID V1 by Perez-Llorens, Albero-Albero and
Silvestre-Blanes (2026), DOI `10.17632/db6g85xsyg.1`, CC BY 4.0. The publisher
archive exceeds 500 MB. This repository therefore includes a download script,
the immutable assessed split, and an authentic 15-patch/8-mask sample covering
all seven publisher origins, both labels, every split, and the app examples.

```bash
PYTHONPATH=src python scripts/download_data.py \
  --url '<V1 HTTPS download URL>' \
  --expected-sha256 f6b4f82cce2f0220bbab3968a74b5337ebf2bbfd5e33d0548d34e1ebf80f85b9
python -m zipfile -e data/raw/RAW-FABRID-V1.zip data/raw/release
```

The complete assessed split has 13,384 training, 3,472 validation, and 2,996
held-out test patches. Publisher origins, source images, and detected duplicate
components do not cross partitions. The small included sample is only for code
and UI checks; it was not used to calculate reported metrics.

## Reproduce the study

After downloading the full dataset, run the scripts in this order. Training
and the VLM evaluation are computationally expensive. Use a fresh output copy:

```bash
PYTHONPATH=src python scripts/prepare_data.py \
  --release data/raw/release/RAW_FABRID \
  --metadata data/raw/release/RAW_FABRID/RAW_FABRID_Patches_Metadata.csv \
  --patch-column patch_name --source-column parent_raw \
  --source-metadata data/raw/release/RAW_FABRID/RAW_FABRID_HighRes_Metadata.csv \
  --area-column defect_area
PYTHONPATH=src python scripts/eda.py
PYTHONPATH=src python scripts/train.py ann --epochs 4 --device cpu
PYTHONPATH=src python scripts/train.py svm --start-index 24
PYTHONPATH=src python scripts/select_models.py
PYTHONPATH=src python scripts/repeat_seeds.py --device cpu
PYTHONPATH=src python scripts/vlm_validate.py --device cpu
PYTHONPATH=src python scripts/xai_validate.py
PYTHONPATH=src python scripts/make_figures.py
PYTHONPATH=src python scripts/freeze.py
PYTHONPATH=src python scripts/final_evaluate.py
```

Model, threshold, prompt, and hyperparameter selection use only training and
validation data. Do not rerun held-out evaluation for tuning. Four executed
notebooks contain saved outputs for dataset/EDA, ANN optimisation, deployment
and XAI, and the VLM/final protocol.

## Submitted evidence

- `src/`, `scripts/`, and `notebooks/`: implementation and executed workflow.
- `app/`, `artifacts/serving-bundle.json`, and
  `artifacts/ann-search-018.pt`: working frontend/backend and selected ANN.
- `data/manifests/`: frozen split and leakage audit.
- `reports/experiment-log.csv`: 54 completed experiment records, including
  unsuccessful attempts.
- `reports/final-evaluation/`: shared held-out ANN/SVM/VLM and XAI evidence.
- `reports/final_report.pdf`: 24 main pages plus two appendices.
- `reports/demonstration.mp4`: 130-second live prediction/XAI/error demo.
- `docs/requirements-matrix.md`: PDF requirement-to-evidence checklist.
- `docs/simplification-summary.md`: exact record of this reduction.

The final ANN has 1,518,881 parameters and a validation-selected threshold of
`0.895234`. Its held-out defect-class F2 is `0.6883`.

## Required disclosures

Lecturer approval correspondence and the live defence remain student-owned
requirements. The first held-out result files were lost and the identical
frozen evaluation was physically rerun; this is disclosed in the report and
requires lecturer judgement against the brief's once-only rule. No adverse
classical, VLM, XAI, or incorrect-prediction result has been removed.
