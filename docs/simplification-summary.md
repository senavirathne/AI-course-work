# Simplification summary

This is the reduced coursework version of the fabric-defect project. It was audited directly against the supplied ten-page **AI Coursework 2025 Batch** brief. It keeps the files needed to demonstrate each assessed requirement and omits development-only, duplicated, or packaging-only material.

## Implementation changes

- Reduced the deliverable at the project/file level instead of rewriting the evidence-producing scientific code after final evaluation.
- Removed the unused test and report-builder dependency groups and configuration because those omitted utilities are not coursework deliverables.
- Kept one environment definition for the core application plus only the VLM and notebook extras required to reproduce Questions 1–6.
- Kept the real-data checks, validation/test isolation, input validation, experiment logging, Grad-CAM validation, and VLM protocol because the brief explicitly requires them.
- Kept the model architecture, preprocessing, experiment configurations, metrics, selected checkpoint, and saved results unchanged.

## What is retained

- `README.md`, `pyproject.toml`, and `requirements.txt` for setup and reproduction.
- `src/` and `scripts/` for data preparation, ANN/SVM training, model selection, evaluation, explainability, and VLM comparison.
- `app/` plus `artifacts/ann-search-018.pt` and `artifacts/serving-bundle.json` for request-time trained-model inference and Grad-CAM.
- Four executed notebooks for the assessed walkthrough.
- The authentic representative dataset sample, fixed split manifests, provenance audit, and download script. The publisher archive is over 500 MB and is therefore not included.
- The selected experiment records, complete experiment table, final comparison, error analysis, XAI evidence, VLM evidence, and required report figures.
- `reports/final_report.pdf` and the 130-second `reports/demonstration.mp4`.
- The AI-use declaration, protocol records, requirements matrix, and disclosed held-out-evaluation recovery exception.

## What is omitted

- Synthetic development data and development-run outputs.
- Non-selected model checkpoints and duplicated prediction exports.
- The original automated test suite, report-building utilities, recording internals, screenshots, transcripts, and repeated verification reports.
- Repository-restore helpers, split archive parts, submission ZIP parts, and other packaging machinery.
- Markdown copies of evidence already present in the final PDF or structured JSON/CSV files.

These omissions do not replace assessed results. The retained `reports/experiment-log.csv` still includes successful and unsuccessful configurations, while the retained JSON evidence contains the measured comparisons used by the report.

## Requirement coverage

The direct mapping from each coursework requirement to a retained file is in `docs/requirements-matrix.md`. The only non-complete items are already disclosed there:

- lecturer approval is student-confirmed, but the student must retain the original correspondence;
- the held-out protocol was physically rerun after its first result files were lost; and
- personal understanding and the live individual defence must be completed by the student.

No software edit can resolve these items, so they are not presented as complete.

## Verification performed on the reduced version

- All retained Python files compile successfully.
- All four notebooks are valid JSON, contain saved outputs, and contain no saved exception output.
- The retained JSON files parse and the CSV files have consistent row widths.
- All 168 representative patches and 84 masks match the SHA-256 values in the sample manifest.
- The model bundle points to the included checkpoint, whose SHA-256 matches the bundle.
- Every demonstration path in `data/demo/cases.json` resolves to an included sample image or mask.
- The report has 26 PDF pages; Appendix A starts on page 25, leaving 24 assessed main pages.
- The demonstration MP4 is playable and 130.040 seconds long, below the 480-second limit.
- The application could not be executed in the simplification environment because its third-party Python packages were not installed. A previously measured passing application run remains in `reports/application-verification/results.json`; rerun the application after following the README setup commands.

## Result

The reduced worktree is the submission-oriented project. The sibling `AI-course-work` repository was not changed.
