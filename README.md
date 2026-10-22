# Risk-aware, explainable fabric defect detection

This coursework supports a textile Quality Assurance Inspector with patch-level defect classification, inspection-priority guidance, and Grad-CAM explanations. The assessed study uses RAW-FABRID V1 and a single frozen source/origin-grouped train/validation/test split.

## Environment

The project requires Python 3.11 or newer. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[vlm,notebooks]'
```

`requirements.txt` contains the core pinned runtime dependencies. `pyproject.toml` defines the local package and the additional VLM/notebook dependencies used by the submitted work.

The live browser verification additionally requires Node.js, `playwright@1.62.0`, and a Playwright Chromium installation.

## Run the required web application

The selected trained ANN bundle and checkpoint are included. Every request performs inference at request time.

```bash
FABRIC_BUNDLE=artifacts/serving-bundle.json \
FABRIC_CPU_THREADS=1 \
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000> and upload a 256×256 PNG or JPEG from `data/sample/MVTec/test/`. The UI provides the model score, inspection recommendation, limitations, and Grad-CAM explanation. Scores are uncalibrated model scores and must not be interpreted as guaranteed confidence.

## Dataset and fixed split

Dataset: RAW-FABRID V1, Pérez-Llorens, Albero-Albero and Silvestre-Blanes (2026), DOI `10.17632/db6g85xsyg.1`, licensed CC BY 4.0.

The publisher archive exceeds 500 MB, so this submission contains:

- `scripts/download_data.py`, which records the exact publisher URL, version, checksum, licence, and provenance;
- `data/sample/`, a representative authentic sample covering all seven publisher origins;
- `data/sample-manifests/`, which maps and verifies the sample files; and
- `data/manifests/`, the immutable assessed split and leakage audit.

Download and extract the complete dataset before running full training or evaluation:

```bash
PYTHONPATH=src python scripts/download_data.py \
  --url '<exact HTTPS Download All URL from the V1 publisher page>' \
  --expected-sha256 f6b4f82cce2f0220bbab3968a74b5337ebf2bbfd5e33d0548d34e1ebf80f85b9
python -m zipfile -e data/raw/RAW-FABRID-V1.zip data/raw/release
```

The assessed split contains 13,384 training, 3,472 validation, and 2,996 held-out test patches. Related publisher origins, source images, and duplicate components do not cross partitions. The manifest files record the source schema, assignments, and leakage audit.

## Reproducible scientific workflow

The required entry points are:

```bash
PYTHONPATH=src python scripts/eda.py
PYTHONPATH=src python scripts/train.py ann --epochs 4 --device cpu
PYTHONPATH=src python scripts/train.py svm --start-index 24
PYTHONPATH=src python tools/refine_ann.py --device cpu
PYTHONPATH=src python scripts/select_models.py
PYTHONPATH=src python scripts/repeat_seeds.py --device cpu
PYTHONPATH=src python scripts/vlm_validate.py --device cpu
PYTHONPATH=src python scripts/xai_validate.py
PYTHONPATH=src python scripts/make_figures.py
PYTHONPATH=src python tools/summarize_experiments.py
PYTHONPATH=src python scripts/freeze.py
PYTHONPATH=src python scripts/final_evaluate.py
PYTHONPATH=src python tools/finish_evidence.py --post-test
PYTHONPATH=src python tools/verify_live_application.py --cases data/demo/cases.json
```

Run the workflow in a fresh copy/output location. Training and foundation-model validation are computationally expensive. Model selection, thresholds, prompts, and hyperparameters use training/validation data only; the held-out test set is opened by the final evaluation step after the protocol is frozen.

The delivered held-out evaluation was physically computed twice because a workspace reset destroyed the first result files. The identical frozen protocol was rerun without changing the model, split, prompt, or thresholds. This disclosed exception is recorded in `docs/engineering-runs/final-results-workspace-reset.json` and requires assessor judgement against the brief's once-only checklist item.

`tools/reproduce_classical_pilots.py` replays the four retained historical SVM pilot configurations into a separate output directory when those records also need to be regenerated.

The four executed notebooks provide the assessed walkthrough:

1. dataset, EDA, leakage checks, and modelling constraints;
2. ANN/classical experiments, optimisation, failures, and error analysis;
3. serving, Grad-CAM, occlusion checks, and incorrect predictions; and
4. foundation-model validation and the frozen final protocol.

They can be opened and executed directly in Jupyter after installing the notebook dependencies above.

Requirement-by-requirement study guides:

- `docs/question-1-learning-material.md`: dataset, EDA, leakage, metric, and modelling constraints.
- `docs/question-2-learning-material.md`: ANN architecture, design justifications, alternatives, and prior expectation.
- `docs/question-3-learning-material.md`: model development, baselines, learning curves, error analysis, optimisation, and failed approaches.
- `docs/question-4-learning-material.md`: functional frontend/backend deployment, trained-model inference, input testing, and limitations.
- `docs/question-5-learning-material.md`: explainability alternatives, Grad-CAM, sanity checks, mask correspondence, and genuine model errors.
- `docs/question-6-learning-material.md`: genuine VLM evaluation, prompt selection, ANN comparison, deployment trade-offs, and a safe extension.

## Mandatory deliverables included

- `src/`, `scripts/`, and the six retained `tools/` helpers: preprocessing, training, evaluation, experiment logging, application verification, XAI, GenAI, and deployment code.
- `notebooks/`: four executed coursework notebooks with their evidence embedded in the saved outputs.
- `app/`: required frontend and backend.
- `artifacts/serving-bundle.json` and `artifacts/ann-search-018.pt`: selected model required by the application.
- `reports/experiment-log.csv`: complete substantive experiment table, including unsuccessful runs.
- `docs/`: the minimal protocol and evidence records required by the notebooks, frozen evaluation, and report traceability.
- `data/`: reproducible download information, representative sample, fixed split, and leakage audit.
- `reports/demonstration.mp4`: 130-second demonstration showing request-time prediction, explanations, and an incorrect prediction.
- `reports/final_report.pdf`: final Questions 1–6 report with appendices, including the generative-AI-use declaration.
- `docs/simplification-summary.md`: a record of what was retained, removed, and verified in this reduced version.

The final ANN has 1,518,881 parameters and uses a validation-selected threshold of `0.895234`. The report documents predictive, calibration, dataset, XAI, privacy, and deployment limitations. This is coursework evidence, not a production-safety certification.
