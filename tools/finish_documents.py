"""Update the README, evidence matrix and viva notes from completed artifacts."""

from pathlib import Path

from fabric_inspector.common import read_json, now


def main():
    comparison = read_json(Path("reports/final-evaluation/comparison.json"))
    assert comparison["status"] == "COMPLETE"
    bundle = read_json(Path("artifacts/serving-bundle.json"))
    final = read_json(Path("experiments") / (bundle["run_id"] + ".json"))
    checks = read_json(Path("reports/execution-checks.json"))
    app = read_json(Path("reports/application-verification/results.json"))
    video = read_json(Path("reports/demonstration-metadata.json"))
    assert app["status"] == "PASS" and video["status"] == "PASS"
    results = {r["model"]: r for r in comparison["comparisons"]}
    lines = [
        "# Risk-aware, explainable fabric defect detection",
        "",
        "The approved application supports a textile Quality Assurance Inspector with patch-level classification, inspection priority and actual Grad-CAM. Lecturer approval was confirmed by the student. All assessed results use the complete publisher RAW-FABRID V1 release and one frozen source/origin-grouped supervised split. The identical held-out evaluation was physically computed twice after result files were lost; this exception is disclosed below.",
        "",
        "## Git checkout and large files",
        "",
        "Repository: https://github.com/senavirathne/AI-course-work. After cloning, run `python3 tools/restore_repository_assets.py` before reviewing the notebooks or full evaluation evidence. The selected SVM checkpoint and full split CSV are stored in numbered raw-byte parts under `repository-assets/`; the restore command checks every part and each reassembled SHA-256. All other necessary weights, experiment records, raw VLM responses, sample images and the video are tracked directly. A portable submission ZIP already contains the restored originals. `.gitignore` excludes downloaded raw data, environments, caches, credentials and internal browser recordings.",
        "",
        "## Quick verification and application",
        "",
        "Tested: Linux x86_64, Python 3.12.14, CPU-only PyTorch 2.9.1, torchvision 0.24.1. Install from the extracted project root:",
        "",
        "```bash\npython3 -m venv .venv\nsource .venv/bin/activate\npython -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.9.1 torchvision==0.24.1\npython -m pip install -e '.[test,vlm,notebooks]'\npython -m pytest -q\nFABRIC_BUNDLE=artifacts/serving-bundle.json FABRIC_CPU_THREADS=1 python -m uvicorn app.main:app --host 127.0.0.1 --port 8000\n```",
        "",
        "Open http://127.0.0.1:8000 and upload one of the genuine `data/demo/` images listed in `data/demo/cases.json`, or a 256x256 L/RGB PNG/JPEG from `data/sample/`. Mask files are annotations, not intended app input. The selected trained checkpoint is included; the full archive and VLM weights are not needed for this quick check. Every request executes fresh inference. A missing/corrupted checkpoint returns 503. Scores are uncalibrated; routine inspection never guarantees defect-free fabric.",
        "",
        "The four submitted notebooks contain executed outputs. To rerun their inspections (no training/test reselection):",
        "",
        "```bash\npython tools/execute_notebooks.py\n```",
        "",
        "Other platforms need compatible wheels; no untested platform is claimed verified. `requirements-tested.txt` records the actual executed dependency versions.",
        "",
        "## Normal-only publisher training folder: resolved",
        "",
        "The publisher `train/good` folder is an anomaly-detection convention and contains no positive supervised training labels. This project pools all labelled patches from publisher train/test folders and creates a new frozen supervised split. Explicit CSV joins map each patch to its parent image and the publisher `roll_id` fabric-origin field. Related origins, sources and duplicates stay together. Defective patches enter training through this protocol; the publisher anomaly split is not reported as our benchmark.",
        "",
        "| Partition | Publisher origins | Patches | Defective |\n| --- | --- | ---: | ---: |\n| Train | Rollo2A, Rollo4A, Rollo5A, Rollo7A | 13,384 | 539 |\n| Validation | Rollo6A | 3,472 | 52 |\n| Test | Rollo1A, Rollo3A | 2,996 | 96 |",
        "",
        "There are 709 source images and 19,852 patches (19,165 normal / 687 defective). Origin IDs do not prove individual physical rolls, manufacturers or fabric types. Only seven origins limit generalisation. The split SHA-256 is `af7daf13d0854ceeba004f1a06182bac778ac716021dec3d8fc40edf265aa76f`.",
        "",
        "The audit finds 471 patch masks with intermediate grayscale edges. Exactly-white pixels reproduce all 687 publisher CSV defect areas. Masks and labels use this documented convention; raw bytes are unchanged. A normal patch is not assigned its defective parent label, and missing masks are not assumed normal.",
        "",
        "## Official archive and authentic sample",
        "",
        "RAW-FABRID V1, Perez-Llorens / Albero-Albero / Silvestre-Blanes (2026), DOI 10.17632/db6g85xsyg.1, CC BY 4.0: https://data.mendeley.com/datasets/db6g85xsyg/1. The complete 4,133,046,386-byte archive was acquired, CRC-checked and verified against SHA-256 `f6b4f82cce2f0220bbab3968a74b5337ebf2bbfd5e33d0548d34e1ebf80f85b9`.",
        "",
        "Because the data exceed 500 MB, the submission contains the checksum-verified download procedure plus 168 authentic patches and 84 masks from all seven origins. The sample oversamples positives for engineering tests; full-population metrics never use its class balance. Its assignments are copied from the frozen full split.",
        "",
        "```bash\npython tools/download_raw_fabrid.py\npython -m zipfile -e data/raw/RAW-FABRID-V1.zip data/raw/release\n```",
        "",
        "Full manifest, import audit and source checksums are included in `data/manifests/`. For a separately regenerated split, use `scripts/prepare_data.py --help` and the exact publisher metadata fields in `data/README.md`; never overwrite the assessed manifest.",
        "",
        "## Complete real study and reproduction",
        "",
        "The measured search contains 20 ANN screening configurations at up to four epochs, one validation-selected 12-epoch refinement, 24 main classical configurations plus four retained earlier pilots, and selected-model seeds 42/123/2026. Learning rate, batch size, actual depth and two optimisers are compared empirically. All four required scientific unsuccessful attempts, including two substantive approaches, have completed run IDs and bounded diagnoses in `docs/failed-approaches.json`.",
        "",
        "The local VLM uses genuine SmolVLM-256M-Instruct at revision `7e3e67edbbed1bf9888184d9df282b700a323964`, bfloat16 CPU, 256 pixels, no splitting, matching 16 visual tokens and greedy generation. Two prompts use every validation patch. The selected prompt and all final comparisons use every one of the 2,996 frozen test examples. Raw outputs, retries and unresolved decisions are retained. Labels/masks/filenames/ANN scores are never sent to VLM inference. No VLM probability or ranking score is invented. Pretrained weights download from the publishers on first use.",
        "",
        "A temporary-runtime reset during a user pause lost the first completed final-evaluation files and recording. The identical frozen protocol was physically recomputed to reconstruct these artifacts: no training, model selection, cutoff, prompt or split changed. Every delivered prediction is from actual reconstructed inference, with original observed confusion counts used only for cross-checks. `docs/engineering-runs/final-results-workspace-reset.json` records two physical computations. The brief requires evaluating the final models only once; this reconstruction is a disclosed exception requiring lecturer assessment. Acceptance of that exception is not asserted. Delivered timings are freshly measured, and the video is a new actual recording.",
        "",
        "A fresh full reproduction is expensive on CPU and can take several hours. It shares the verified raw release and immutable split, writes all new results into an empty folder and freezes its own evaluation once:",
        "",
        "```bash\npython tools/reproduce_study.py --output ../fabric-reproduction/run-001\n```",
        "",
        "Use `--validation-only` to reproduce the selection stages without opening a new held-out evaluation. The helper checks that historical failure conclusions actually reproduce; if they change, it preserves the new measured results and stops before copying an unsupported diagnosis into a frozen study. The original submitted artifacts are never overwritten.",
        "",
        "The four included notebooks are already executed. To execute their review cells again without retraining or repeating the held-out study: `python tools/execute_notebooks.py`. This uses a separate IPython process per notebook, executes every actual code cell and captures rich outputs. Standard nbclient/Jupyter execution is available with `--engine kernel`; the hosted build environment blocks its interface-discovery socket, so that transport is not claimed tested successfully here. To rebuild the report: `python -m pip install -e '.[report]'`, then `python tools/build_report.py`. Report figures and analyses read saved immutable outputs; they do not choose new models.",
        "",
        "Optional live browser/video verification needs Node.js, `playwright@1.62.0`, a Chromium installation and FFmpeg on PATH. Install Playwright locally with `npm install --no-save playwright@1.62.0` and `npx playwright install chromium`, then run `python tools/verify_live_application.py --cases data/demo/cases.json --record` and `python tools/make_demonstration.py`. Chromium system libraries are platform-specific. These recording prerequisites are separate from the FastAPI application's ordinary browser use.",
        "",
        "Original scientific entry points: `scripts/train.py`, `tools/refine_ann.py`, `scripts/select_models.py`, `scripts/repeat_seeds.py`, `scripts/vlm_validate.py`, `scripts/freeze.py`, `scripts/final_evaluate.py`. Existing run IDs and test-access guards cannot be silently reused. `--resume` skips unchanged completed ANN/SVM records only. Engineering interruptions, solver amendments and source versions are retained separately.",
        "",
        "## Results and deliverables",
        "",
        "| Comparison | Test F2 | TP / FP / FN |\n| --- | ---: | --- |",
    ]
    for name in [
        "always_normal",
        "always_defective",
        "classical",
        "initial_ann",
        "final_ann",
        "vlm",
    ]:
        m = results[name]["metrics"]
        lines.append(f"| {name} | {m['f2']:.4f} | {m['tp']} / {m['fp']} / {m['fn']} |")
    lines.extend(
        [
            "",
            f"Selected ANN: `{bundle['run_id']}`, {final['architecture']['total_parameters']:,} parameters, frozen cutoff {bundle['classification_threshold']:.6f}. The report explains predictive weaknesses, origin/mask-size errors, uncalibrated scores and the assumed recommendation policy. It does not certify production safety.",
            "",
            "- `reports/final_report.pdf` and `.md`: Questions 1-6, real numbers, learning/optimisation evidence and AI-use appendices.\n- `reports/experiment-log.csv` and `.md`, `experiments/`: full measured configurations, histories, lessons and hashes.\n- `reports/final-evaluation/`: frozen-protocol reconstructed outputs, seed repetitions, raw VLM generations and actual XAI.\n- `docs/three-xai-errors.json` and `.md`: three genuine incorrect predictions and measured occlusion.\n- `reports/application-verification/`: real HTTP/browser checks.\n- `reports/demonstration.mp4`, transcript and metadata: actual application recording.\n- `verification.json`, `docs/requirements-matrix.md`: technical completion evidence and disclosed protocol exception.",
            "",
            f"Verification: {checks['pytest_passed']} tests passed; compilation/lint, four notebook executions and live application checks passed. Video duration {video['duration_seconds']:.1f}s (maximum 480s). The student's review, individual understanding and live presentation remain human responsibilities. AI assistance and scripted recording are declared in `docs/ai-use-declaration.md`.",
            "",
            "Historical `development-run/` and `data/development/` are explicitly synthetic engineering checks, excluded from assessed results. Nonselected large SVM artifacts may be omitted from the archive; every run record is included, with retained final/failure-control checkpoints identified in `docs/packaged-artifacts.json`.",
        ]
    )
    Path("README.md").write_text("\n".join(lines) + "\n")
    matrix = [
        (
            "Q1: source/licence/size/target",
            "data/README.md; data/manifests/data-audit.json; report Q1",
        ),
        (
            "Q1: preprocessing, leakage and groups",
            "data/manifests/split.json / split.csv; shared data/images.py; integrity tests",
        ),
        (
            "Q1: EDA and three modelling constraints",
            "docs/eda-statistics.json / eda-modelling-findings.json; real figures",
        ),
        (
            "Q1: consequences and primary metric",
            "report Q1; evaluation/metrics.py; validation-only F2 selection",
        ),
        (
            "Q2.a: layers, dimensions, parameters and diagram",
            "actual architecture CSVs and run JSON; report Q2.a",
        ),
        (
            "Q2.a: separate design/loss justifications",
            "report Q2.a; same preprocessing in training/API/XAI",
        ),
        (
            "Q2.b: alternatives, ANN suitability and prior hypothesis",
            "report Q2.b; docs/pre-evaluation-hypothesis.md",
        ),
        (
            "Q3.a: initial/final ANN and learning curves",
            "all 21 ANN candidate records; reports/figures/initial-final-learning.png",
        ),
        (
            "Q3.a: comparable classical tuning",
            "24 main configurations + four retained pilots; reports/experiment-log.csv",
        ),
        (
            "Q3.a: references, final metrics and latency",
            "reports/final-evaluation/comparison.json and per-model outputs",
        ),
        (
            "Q3.a: meaningful full-population error analysis",
            "reports/error-analysis.json; origin/mask-size/recommendation tables",
        ),
        (
            "Q3.b: three hyperparameters and two optimisers",
            "reports/optimisation-analysis.json; matched screening runs; report Q3.b",
        ),
        (
            "Four unsuccessful attempts; two substantive approaches",
            "docs/failed-approaches.json; completed run IDs and controls",
        ),
        (
            "Q4: functional trained request-time ANN frontend/backend",
            "app/; actual bundle/checkpoint; reports/application-verification/results.json",
        ),
        (
            "Q4: score meaning, limitations, normal/invalid/OOD testing",
            "UI/API labels; 13 actual HTTP checks; browser checks; unit tests",
        ),
        (
            "Q5.a: three methods, cost/limitations and justified choice",
            "report Q5.a: Grad-CAM, Integrated Gradients, LIME",
        ),
        (
            "Q5.b: actual explanation UI and sanity checks",
            "reports/xai-validation/; final-evaluation/xai/; matched-area top/low/random occlusion",
        ),
        (
            "Three genuine incorrect predictions analysed with XAI",
            "docs/three-xai-errors.json; three inspected original/mask/overlay panels",
        ),
        (
            "Q6.a: real VLM, exact prompts and shared examples",
            "experiments/vlm-validation-{0,1}.json; final-evaluation/vlm.json; immutable revision",
        ),
        (
            "Q6.b: numerical comparison, costs/privacy/deployment/extension",
            "report Q6.b; actual token/latency/failure logs",
        ),
        (
            "Single split, validation-only selection, seeds and complete logs",
            "final-evaluation-manifest.json; exclusive COMPLETE test-access guard; three frozen seeds",
        ),
        (
            "Code, environment, executed notebooks and README",
            "pinned pyproject/requirements-tested; four executed notebooks; verification.json",
        ),
        (
            "Dataset >500 MB: download script and representative sample",
            "verified publisher downloader; authentic 168 patches/84 masks; sample checksums",
        ),
        ("Report <=25 main pages", "reports/final_report.pdf; reports/pdf-verification.json"),
        (
            "Actual demonstration <=8 minutes",
            "reports/demonstration.mp4; duration/decode checks; genuine incorrect prediction",
        ),
        ("AI-use appendix", "docs/ai-use-declaration.md; report appendix"),
    ]
    text = [
        "# Coursework requirement matrix",
        "",
        "The supplied ten-page coursework brief is authoritative. The student confirmed lecturer approval of the proposal. COMPLETE refers to delivered and verified technical evidence; it does not assert individual understanding or a completed live presentation.",
        "",
        "| Requirement | Actual evidence | Status |",
        "| --- | --- | --- |",
    ]
    text.extend(f"| {criterion} | {evidence} | COMPLETE |" for criterion, evidence in matrix)
    text.extend(
        [
            "| Strictly one physical held-out evaluation | Two identical frozen computations after result loss; docs/engineering-runs/final-results-workspace-reset.json | DISCLOSED EXCEPTION; lecturer acceptance unverified |",
            "| Individual review, understanding and live defence | Student review and live presentation; docs/viva-guide.md | HUMAN TASK |",
            "",
            f"Updated {now()}. Real results and adverse findings are retained; weak performance is not concealed.",
        ]
    )
    Path("docs/requirements-matrix.md").write_text("\n".join(text) + "\n")
    Path("docs/progress.md").write_text(
        f"# Technical submission status\n\nAll technical coursework deliverables are complete and verified in verification.json. {checks['pytest_passed']} tests passed; four notebooks and live HTTP/browser checks passed. Full test evaluation status COMPLETE, same 2,996 examples for every comparison. Selected ANN {bundle['run_id']}. Three genuine incorrect predictions are analysed with XAI. Actual video duration {video['duration_seconds']:.1f}s.\n\nThe final report includes all six questions, four actual unsuccessful attempts including two substantive approaches, and honest comparison/limitations. Full raw data are supplied through a verified download script plus an authentic engineering sample.\n\nThe frozen evaluation was physically computed twice to reconstruct lost files. This is an exception to the brief's once-only rule; lecturer acceptance has not been verified. Technical completion is not a claim of unconditional coursework compliance. The student's personal review and individual live presentation remain human tasks.\n"
    )
    guide = f"""# Individual viva preparation

Read the report, run the app and inspect the actual code. These notes do not demonstrate your understanding for you.

1. Explain the specific user: textile Quality Assurance Inspector. A patch score changes inspection priority; the inspector makes the release/hold decision.
2. Explain why publisher train/good alone is unsuitable for supervised binary training. We pool labelled patches, then freeze source/duplicate/publisher-origin groups before any fit. This is a new supervised benchmark, not the publisher anomaly split.
3. State the full counts: 19,852 patches, 687 positive; train 13,384/539, validation 3,472/52, test 2,996/96. Seven anonymised origin IDs do not establish seven independently known physical rolls/manufacturers.
4. Explain exact-white masks: 471 masks have intermediate gray edges; intensity 255 reproduces all CSV areas. Do not copy parent labels or assume absent masks mean normal.
5. Derive F2 = 5TP/(5TP+FP+4FN), explain recall emphasis and why accuracy alone is weak. Validation cutoffs are frozen. AP needs continuous scores; the label-only VLM has no invented AP/AUC/probability.
6. Describe the initial 5,977-parameter CNN: Conv3x3/BN/ReLU/Pool blocks at widths 8/16/32; GAP/dropout 0.2/linear logit. Explain local texture assumptions, tiny-defect dilution, normalisation, train-derived positive weight 23.8312 and BCEWithLogitsLoss.
7. Describe final {bundle["run_id"]}: {final["config"]["model"]}, {final["architecture"]["total_parameters"]:,} parameters. Inspect its actual architecture CSV and checkpoint metadata; do not repeat an obsolete width-16 parameter count.
8. Trace one controlled learning-rate, batch-size and actual-depth comparison. Fixed epochs are not fixed update counts. Compare AdamW and momentum SGD at their own tested rates; four epochs do not establish eventual optimiser superiority.
9. Defend the classical budget: 24 main LBP/HOG+LBP, linear/RBF, C .1/1/10, class weights None/balanced configurations; four historical pilots. Explain train-only scaler, the LinearSVC squared-hinge versus libsvm pilot distinction, and the convergence-only numerical fallback that preserves the original solver budget.
10. Explain all four scientific failures from actual completed run IDs. Two are substantive descriptor/whole-approach failures; two are learning-rate attempts. Engineering interruptions and synthetic checks do not count.
11. State actual final ANN F2 {results["final_ann"]["metrics"]["f2"]:.4f} versus classical {results["classical"]["metrics"]["f2"]:.4f} and VLM {results["vlm"]["metrics"]["f2"]:.4f}. Explain the pre-evaluation conditional ANN expectation honestly. No test score changed selection.
12. Interpret learning curves and seed spread. Weighted loss gaps also reflect prevalence differences. Two held-out origins make bootstrap intervals coarse; patches are not independent factory samples.
13. Launch the app and upload data/demo cases. Explain the checkpoint hash, fresh request-time prediction, uncalibrated score, frozen class cutoff, separate recommendation costs and limitations. Discuss normal, invalid, blank and noise checks; reliable OOD detection is not implemented.
14. Explain Grad-CAM's defect-logit target and chosen convolutional layer. Compare Integrated Gradients and LIME. Check top/low/random 10% mean-fill occlusion: negative drops mean increased defect score. A zero CAM has no validated hotspot. Map overlap and attractive overlays are not causal proof.
15. Show three genuine errors with original, mask and overlay, their actual scores/cutoff and measured perturbations. Do not invent a manufacturing cause or severity.
16. Explain pinned SmolVLM revision, bfloat16/256 pixels/16 visual tokens/no splitting/greedy 24-token decoding/one retry. Labels, masks, filenames and ANN outputs do not reach inference. Ambiguous responses remain unresolved in all denominators.
17. Discuss local compute/download cost and privacy, label-format failures, compressed tiny detail, possible unknown pretrained-data overlap and a bounded inspection-note extension.
18. Declare the ChatGPT/Codex assistance and scripted real browser video. Review code/results personally and give the individual live presentation under institutional requirements.

Useful evidence: README.md; docs/requirements-matrix.md; reports/final_report.pdf; reports/experiment-log.csv; artifacts/final-evaluation-manifest.json; reports/final-evaluation/; docs/three-xai-errors.json; verification.json.
"""
    Path("docs/viva-guide.md").write_text(guide)
    print("Updated evidence-based README, matrix, status and viva guide")


if __name__ == "__main__":
    main()
