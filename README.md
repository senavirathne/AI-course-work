# Risk-aware, explainable fabric defect detection

The approved application supports a textile Quality Assurance Inspector with patch-level classification, inspection priority and actual Grad-CAM. Lecturer approval was confirmed by the student. All assessed results use the complete publisher RAW-FABRID V1 release and one frozen source/origin-grouped supervised split. The identical held-out evaluation was physically computed twice after result files were lost; this exception is disclosed below.

## Git checkout and large files

Repository: https://github.com/senavirathne/AI-course-work. After cloning, run `python3 tools/restore_repository_assets.py` before reviewing the notebooks or full evaluation evidence. The selected SVM checkpoint and full split CSV are stored in numbered raw-byte parts under `repository-assets/`; the restore command checks every part and each reassembled SHA-256. All other necessary weights, experiment records, raw VLM responses, sample images and the video are tracked directly. A portable submission ZIP already contains the restored originals. `.gitignore` excludes downloaded raw data, environments, caches, credentials and internal browser recordings.

## Quick verification and application

Tested: Linux x86_64, Python 3.12.14, CPU-only PyTorch 2.9.1, torchvision 0.24.1. Install from the extracted project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.9.1 torchvision==0.24.1
python -m pip install -e '.[test,vlm,notebooks]'
python -m pytest -q
FABRIC_BUNDLE=artifacts/serving-bundle.json FABRIC_CPU_THREADS=1 python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000 and upload one of the genuine `data/demo/` images listed in `data/demo/cases.json`, or a 256x256 L/RGB PNG/JPEG from `data/sample/`. Mask files are annotations, not intended app input. The selected trained checkpoint is included; the full archive and VLM weights are not needed for this quick check. Every request executes fresh inference. A missing/corrupted checkpoint returns 503. Scores are uncalibrated; routine inspection never guarantees defect-free fabric.

The four submitted notebooks contain executed outputs. To rerun their inspections (no training/test reselection):

```bash
python tools/execute_notebooks.py
```

Other platforms need compatible wheels; no untested platform is claimed verified. `requirements-tested.txt` records the actual executed dependency versions.

## Normal-only publisher training folder: resolved

The publisher `train/good` folder is an anomaly-detection convention and contains no positive supervised training labels. This project pools all labelled patches from publisher train/test folders and creates a new frozen supervised split. Explicit CSV joins map each patch to its parent image and the publisher `roll_id` fabric-origin field. Related origins, sources and duplicates stay together. Defective patches enter training through this protocol; the publisher anomaly split is not reported as our benchmark.

| Partition | Publisher origins | Patches | Defective |
| --- | --- | ---: | ---: |
| Train | Rollo2A, Rollo4A, Rollo5A, Rollo7A | 13,384 | 539 |
| Validation | Rollo6A | 3,472 | 52 |
| Test | Rollo1A, Rollo3A | 2,996 | 96 |

There are 709 source images and 19,852 patches (19,165 normal / 687 defective). Origin IDs do not prove individual physical rolls, manufacturers or fabric types. Only seven origins limit generalisation. The split SHA-256 is `af7daf13d0854ceeba004f1a06182bac778ac716021dec3d8fc40edf265aa76f`.

The audit finds 471 patch masks with intermediate grayscale edges. Exactly-white pixels reproduce all 687 publisher CSV defect areas. Masks and labels use this documented convention; raw bytes are unchanged. A normal patch is not assigned its defective parent label, and missing masks are not assumed normal.

## Official archive and authentic sample

RAW-FABRID V1, Perez-Llorens / Albero-Albero / Silvestre-Blanes (2026), DOI 10.17632/db6g85xsyg.1, CC BY 4.0: https://data.mendeley.com/datasets/db6g85xsyg/1. The complete 4,133,046,386-byte archive was acquired, CRC-checked and verified against SHA-256 `f6b4f82cce2f0220bbab3968a74b5337ebf2bbfd5e33d0548d34e1ebf80f85b9`.

Because the data exceed 500 MB, the submission contains the checksum-verified download procedure plus 168 authentic patches and 84 masks from all seven origins. The sample oversamples positives for engineering tests; full-population metrics never use its class balance. Its assignments are copied from the frozen full split.

```bash
python tools/download_raw_fabrid.py
python -m zipfile -e data/raw/RAW-FABRID-V1.zip data/raw/release
```

Full manifest, import audit and source checksums are included in `data/manifests/`. For a separately regenerated split, use `scripts/prepare_data.py --help` and the exact publisher metadata fields in `data/README.md`; never overwrite the assessed manifest.

## Complete real study and reproduction

The measured search contains 20 ANN screening configurations at up to four epochs, one validation-selected 12-epoch refinement, 24 main classical configurations plus four retained earlier pilots, and selected-model seeds 42/123/2026. Learning rate, batch size, actual depth and two optimisers are compared empirically. All four required scientific unsuccessful attempts, including two substantive approaches, have completed run IDs and bounded diagnoses in `docs/failed-approaches.json`.

The local VLM uses genuine SmolVLM-256M-Instruct at revision `7e3e67edbbed1bf9888184d9df282b700a323964`, bfloat16 CPU, 256 pixels, no splitting, matching 16 visual tokens and greedy generation. Two prompts use every validation patch. The selected prompt and all final comparisons use every one of the 2,996 frozen test examples. Raw outputs, retries and unresolved decisions are retained. Labels/masks/filenames/ANN scores are never sent to VLM inference. No VLM probability or ranking score is invented. Pretrained weights download from the publishers on first use.

A temporary-runtime reset during a user pause lost the first completed final-evaluation files and recording. The identical frozen protocol was physically recomputed to reconstruct these artifacts: no training, model selection, cutoff, prompt or split changed. Every delivered prediction is from actual reconstructed inference, with original observed confusion counts used only for cross-checks. `docs/engineering-runs/final-results-workspace-reset.json` records two physical computations. The brief requires evaluating the final models only once; this reconstruction is a disclosed exception requiring lecturer assessment. Acceptance of that exception is not asserted. Delivered timings are freshly measured, and the video is a new actual recording.

A fresh full reproduction is expensive on CPU and can take several hours. It shares the verified raw release and immutable split, writes all new results into an empty folder and freezes its own evaluation once:

```bash
python tools/reproduce_study.py --output ../fabric-reproduction/run-001
```

Use `--validation-only` to reproduce the selection stages without opening a new held-out evaluation. The helper checks that historical failure conclusions actually reproduce; if they change, it preserves the new measured results and stops before copying an unsupported diagnosis into a frozen study. The original submitted artifacts are never overwritten.

The four included notebooks are already executed. To execute their review cells again without retraining or repeating the held-out study: `python tools/execute_notebooks.py`. This uses a separate IPython process per notebook, executes every actual code cell and captures rich outputs. Standard nbclient/Jupyter execution is available with `--engine kernel`; the hosted build environment blocks its interface-discovery socket, so that transport is not claimed tested successfully here. To rebuild the report: `python -m pip install -e '.[report]'`, then `python tools/build_report.py`. Report figures and analyses read saved immutable outputs; they do not choose new models.

Optional live browser/video verification needs Node.js, `playwright@1.62.0`, a Chromium installation and FFmpeg on PATH. Install Playwright locally with `npm install --no-save playwright@1.62.0` and `npx playwright install chromium`, then run `python tools/verify_live_application.py --cases data/demo/cases.json --record` and `python tools/make_demonstration.py`. Chromium system libraries are platform-specific. These recording prerequisites are separate from the FastAPI application's ordinary browser use.

Original scientific entry points: `scripts/train.py`, `tools/refine_ann.py`, `scripts/select_models.py`, `scripts/repeat_seeds.py`, `scripts/vlm_validate.py`, `scripts/freeze.py`, `scripts/final_evaluate.py`. Existing run IDs and test-access guards cannot be silently reused. `--resume` skips unchanged completed ANN/SVM records only. Engineering interruptions, solver amendments and source versions are retained separately.

## Results and deliverables

| Comparison | Test F2 | TP / FP / FN |
| --- | ---: | --- |
| always_normal | 0.0000 | 0 / 0 / 96 |
| always_defective | 0.1420 | 96 / 2900 / 0 |
| classical | 0.0623 | 5 / 12 / 91 |
| initial_ann | 0.0000 | 0 / 0 / 96 |
| final_ann | 0.6883 | 68 / 42 / 28 |
| vlm | 0.1391 | 94 / 2900 / 2 |

Selected ANN: `ann-search-018`, 1,518,881 parameters, frozen cutoff 0.895234. The report explains predictive weaknesses, origin/mask-size errors, uncalibrated scores and the assumed recommendation policy. It does not certify production safety.

- `reports/final_report.pdf` and `.md`: Questions 1-6, real numbers, learning/optimisation evidence and AI-use appendices.
- `reports/experiment-log.csv` and `.md`, `experiments/`: full measured configurations, histories, lessons and hashes.
- `reports/final-evaluation/`: frozen-protocol reconstructed outputs, seed repetitions, raw VLM generations and actual XAI.
- `docs/three-xai-errors.json` and `.md`: three genuine incorrect predictions and measured occlusion.
- `reports/application-verification/`: real HTTP/browser checks.
- `reports/demonstration.mp4`, transcript and metadata: actual application recording.
- `verification.json`, `docs/requirements-matrix.md`: technical completion evidence and disclosed protocol exception.

Verification: 52 tests passed; compilation/lint, four notebook executions and live application checks passed. Video duration 130.0s (maximum 480s). The student's review, individual understanding and live presentation remain human responsibilities. AI assistance and scripted recording are declared in `docs/ai-use-declaration.md`.

Historical `development-run/` and `data/development/` are explicitly synthetic engineering checks, excluded from assessed results. Nonselected large SVM artifacts may be omitted from the archive; every run record is included, with retained final/failure-control checkpoints identified in `docs/packaged-artifacts.json`.

## Committed submission ZIP and demonstration

The complete submission ZIP is preserved in [submission/](submission/README.md) as checksum-verified parts. Run `python3 tools/restore_submission_zip.py` to create `submission/AI-course-work-submission.zip`. The [demo video](submission/demonstration.mp4) is committed alongside the archive parts and is also included in the ZIP.
