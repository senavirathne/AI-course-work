# Dataset files and reproduction

The complete RAW-FABRID V1 archive was acquired on 11 October 2026 through its publisher's documented public-file endpoint. Its 4,133,046,386 bytes match publisher SHA-256 `f6b4f82cce2f0220bbab3968a74b5337ebf2bbfd5e33d0548d34e1ebf80f85b9`.

Source: https://data.mendeley.com/datasets/db6g85xsyg/1, DOI 10.17632/db6g85xsyg.1. Attribution: Ruben Perez-Llorens, Teresa Albero-Albero and Javier Silvestre-Blanes (2026). Licence: CC BY 4.0, https://creativecommons.org/licenses/by/4.0/.

| Directory | Purpose |
| --- | --- |
| `manifests/` | Immutable full-data split, original checksums, source/origin grouping and import audit |
| `sample/` | 168 unmodified authentic patches, 84 corresponding raw masks and original publisher metadata |
| `sample-manifests/` | Engineering sample checksums and assignments copied from the full split; oversampled positives are never assessed metrics |
| `demo/` | Actual final-evaluation case files used by the recorded application walkthrough and XAI error analysis, when produced |
| `raw/` | Full downloaded archive/release location, excluded from the submission because the archive exceeds 500 MB; acquisition record retained |
| `development/`, `development-manifests/` | Historical artificial engineering fixture, explicitly excluded from real coursework results |

Run `python tools/download_raw_fabrid.py` from the project root, then `python -m zipfile -e data/raw/RAW-FABRID-V1.zip data/raw/release`. The exact archive contains root folder `RAW_FABRID`; the full manifest therefore resolves `data/raw/release/RAW_FABRID`. Existing archive bytes are checked before reuse. Verified range chunks are resumable; the final complete archive must match the publisher checksum.

The new supervised research split combines all labelled patches. Original `train/good` contains no positives because it serves anomaly detection. Every source image, verified duplicate relationship and publisher `roll_id` origin group stays within one partition. The split hash is `af7daf13d0854ceeba004f1a06182bac778ac716021dec3d8fc40edf265aa76f`.

Patch masks contain grey edge values in 471 cases. Only exact-white pixels (255) are foreground; this reproduces every CSV defect area. Original image/mask bytes are unchanged. Normal labels are established from documented metadata/folders, never inferred from absent masks or defective-parent labels. Classifier normalisation and training flips are documented transformations; masks are excluded from classifier inputs.

See `docs/data-card.md` for all counts, field schemas, acquisition limits, origin semantics and leakage checks. The small sample runs serving, explanation and integrity checks without the full archive. It cannot reproduce assessed performance estimates; those use all 19,852 original patches in the full split.
