# Risk-Aware, Explainable Fabric Defect Detection

The approved task supports a textile Quality Assurance Inspector with patch-level defect classification, inspection priority and Grad-CAM. Lecturer approval was confirmed by the student. The inspector makes the operational decision; the model does not certify a roll or control machinery.

## Environment

Tested on Linux x86_64, Python 3.12, eight allocated CPU cores. Use the pinned environment; pretrained weights require internet on first acquisition. CPU-only installation:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.9.1 torchvision==0.24.1
python -m pip install -e '.[test,vlm]'
```

Other operating systems need compatible package wheels; no Intel macOS installation is claimed to have been verified. The viewer is a standard web browser. `requirements-tested.txt` records the executed environment.

## Dataset and the normal-only publisher folders

The full official RAW-FABRID V1 archive was acquired and verified against publisher SHA-256 `f6b4f82cce2f0220bbab3968a74b5337ebf2bbfd5e33d0548d34e1ebf80f85b9`: 4,133,046,386 bytes. Source: https://data.mendeley.com/datasets/db6g85xsyg/1, DOI 10.17632/db6g85xsyg.1, CC BY 4.0. Attribution: Perez-Llorens, Albero-Albero and Silvestre-Blanes (2026).

The audit verified 709 source images, 19,852 patches (19,165 normal / 687 defective) and 687 corresponding patch masks. Patch metadata fields are `patch_name,parent_raw,type,defect_area`; high-resolution metadata provides `filename,roll_id,has_defect,total_area_px,num_defects`.

The original `train/good` folder supports anomaly detection and has no positive supervised training labels. Our approved supervised protocol combines all labelled patches and freezes a new split before fitting. All images sharing a publisher roll_id origin stay together, as do source and duplicate relationships. This evaluates held-out publisher origin groups and is not a score on the publisher's original anomaly split.

The frozen assignment is:

| Partition | Publisher origins | Patches | Defective |
| --- | --- | ---: | ---: |
| Train | Rollo2A, Rollo4A, Rollo5A, Rollo7A | 13,384 | 539 |
| Validation | Rollo6A | 3,472 | 52 |
| Test | Rollo1A, Rollo3A | 2,996 | 96 |

Split SHA-256: `af7daf13d0854ceeba004f1a06182bac778ac716021dec3d8fc40edf265aa76f`. The small number of publisher origin groups is a major generalisation limitation, despite the large patch count.

471 patch masks contain intermediate grey intensities. Exactly-white pixels (255) reproduce the CSV defect area for all 687 masks. The importer explicitly applies that foreground convention, preserves raw files and records it. This corrects the assumption that every mask contains only 0/255. Defect-free patches remain normal even when their parent has another defective patch.

## Download and reproduce

The archive exceeds 500 MB, so the submission includes a verified download procedure and an authentic 168-patch representative engineering sample covering all seven publisher origin groups. The sample oversamples defects and is not an assessed performance population. Its assignments are copied from the full split. Historical `development-run/` results and 100 artificial development images are retained as labelled engineering history, not coursework metrics.

From the project root, download the exact publisher archive and extract it:

```bash
python tools/download_raw_fabrid.py
python -m zipfile -e data/raw/RAW-FABRID-V1.zip data/raw/release
python scripts/prepare_data.py --release data/raw/release/RAW_FABRID \
  --metadata data/raw/release/RAW_FABRID/RAW_FABRID_Patches_Metadata.csv \
  --patch-column patch_name --source-column parent_raw --area-column defect_area \
  --source-metadata data/raw/release/RAW_FABRID/RAW_FABRID_HighRes_Metadata.csv \
  --output my-reproduction/data/manifests/split.json
```

The submitted full manifest is immutable. The example above writes a new reproduction manifest; never overwrite the submitted assignment. To reproduce its exact paths, reuse `data/manifests/split.json` after extraction.

The predeclared CPU study uses 20 ANN configurations at up to four epochs, one validation-selected 12-epoch refinement, 24 main SVM configurations plus four retained measured pilots, and selected-model seeds 42/123/2026. The same full validation population selects VLM prompts; every final comparison uses all 2,996 test examples. `docs/study-budget.json` records the choices before real training.

```bash
python scripts/eda.py
python scripts/train.py ann --epochs 4 --device cpu
python scripts/train.py svm --start-index 24
python tools/refine_ann.py --device cpu
python scripts/select_models.py
python scripts/repeat_seeds.py --device cpu
python scripts/xai_validate.py
python scripts/vlm_validate.py --device cpu
python scripts/make_figures.py
python tools/summarize_experiments.py
python scripts/freeze.py
python scripts/final_evaluate.py
python -m pytest -q
```

Run searches into a fresh `--output` folder when reproducing; existing run IDs and final test access cannot be silently overwritten. Complete logs, configurations, checkpoint hashes, predictions, timings and interpretations accompany the report. Four genuine unsuccessful experiments must cite actual completed run IDs, including two substantive diagnoses. Synthetic results and execution errors do not satisfy that requirement.

## Application

After the real serving bundle is selected:

```bash
FABRIC_BUNDLE=artifacts/serving-bundle.json FABRIC_CPU_THREADS=1 \
  python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000 and upload a 256x256 PNG/JPEG patch from `data/sample/`. The trained checkpoint is loaded once and each request performs fresh inference. Original and actual Grad-CAM are shown with opacity control, score meaning, inspection recommendation and timings. Invalid input is rejected. Missing/corrupted weights produce HTTP 503; no random model, canned prediction or heatmap is substituted. Blank/noise inputs are tested, and reliable OOD detection is not claimed.

The local VLM uses real SmolVLM-256M-Instruct weights at immutable revision `7e3e67edbbed1bf9888184d9df282b700a323964`, 256-pixel processing without tiling, matching 16 image tokens, bfloat16 CPU computation and greedy generation. It receives pixels without labels, masks, filenames or ANN outputs. Its declared parser accepts only one exact binary label or a canonical one-decision sentence; ambiguous output remains unresolved, is retried at most once and stays in all denominators. Raw responses, tokens and latency are retained. No VLM confidence or class-only AUC is invented.

## Assessment evidence

The report follows Questions 1-6. `docs/requirements-matrix.md` maps each criterion to actual evidence; `verification.json` records technical checks. The four notebooks explain and inspect the executed workflow. The demonstration must be an actual recording of fresh predictions, XAI and a genuine incorrect prediction, under eight minutes. AI assistance is declared separately from the assessed VLM. The student remains responsible for understanding and defending the submitted work.
