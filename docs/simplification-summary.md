# Final reduction record

This tree is the smallest submission-safe version produced from the supplied
ten-page **AI Coursework 2025 Batch** brief. The checkpoint immediately before
this reduction is Git commit `4ba51e3`.

## Final tree

- 103 tracked files, 58.87 MiB.
- 844 tracked files removed from the 947-file checkpoint tree.
- 15 authentic sample patches and eight masks instead of 168 patches and 84
  masks. The reduced sample still covers all seven publisher origins, both
  labels, every assessed split, and all three demonstration cases.
- The full 19,852-row frozen split CSV is retained because deleting or changing
  it would break the recorded split hash and study reproducibility.

## Removed

- synthetic development data and development-run outputs;
- 22 non-selected ANN checkpoints, repeat checkpoints, redundant SVM artifacts,
  and duplicated prediction exports;
- repository-asset chunks, restore material, submission ZIP parts, and duplicate
  copies of the report/video;
- automated tests, browser-recording internals, report/notebook builders,
  packaging tools, recovery tools, and repeated verification reports;
- intermediate figures already embedded in the final PDF, transcripts, Markdown
  report copies, execution logs, screenshots, and redundant learning notes;
- all but the key initial/final ANN and two VLM experiment JSON files;
- duplicated demo images (the demo now uses the representative sample directly).

## Retained because the brief requires them

- `src/` and `scripts/`: data preparation, EDA, ANN/SVM training, selection,
  repeat seeds, XAI, VLM comparison, freezing, and held-out evaluation;
- `notebooks/`: four executed notebooks with saved outputs;
- `app/`: functional frontend and backend performing request-time inference;
- the selected ANN checkpoint/bundle, fixed split, leakage audit, full experiment
  log, key raw comparison evidence, and three-error XAI analysis;
- the representative sample and reproducible download procedure;
- the 24-main-page report and 130-second demonstration video;
- a concise requirement matrix and the necessary approval/recovery disclosures.

## Integrity decision

The evaluated scientific source was not rewritten merely to reduce line count.
After removing one post-evaluation provenance-only field, its current executable
source SHA-256 is
`351c100577969134ae95ac2cd0323b29a5cc84e5787b970ca40c446928506c50`,
exactly matching `artifacts/final-evaluation-manifest.json`. A smaller rewritten
model pipeline would no longer be the implementation that produced the reported
results and would require retraining and a new held-out study.

## Remaining non-software obligations

- The student must retain the lecturer's real approval correspondence.
- The live presentation and defence must be completed by the student.
- The identical frozen held-out protocol was physically run twice after the
  first result files were lost. The exception remains disclosed and lecturer
  acceptance is not claimed.

No adverse ANN, classical, VLM, XAI, invalid-input, OOD, or incorrect-prediction
result was removed. Requirement coverage is mapped in
`docs/requirements-matrix.md`.
