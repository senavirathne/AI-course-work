# Question 1 learning material: dataset and exploratory data analysis

This guide explains how the submitted project answers Question 1 of the coursework brief and why each part is required. It is a study aid for understanding and defending the work; the assessed answer itself is on pages 2–4 of `reports/final_report.pdf`.

## What Question 1 requires

Question 1 is worth 10 marks. The brief requires the answer to:

1. describe the dataset source, licence or permission, structure, size, target, and relevant characteristics;
2. perform comprehensive exploratory data analysis appropriate to image data;
3. explain the relative consequences of false positives and false negatives;
4. define and justify the primary evaluation metric using the user's decision and the data characteristics;
5. report leakage checks and the handling of data-quality problems; and
6. finish with exactly three findings that materially constrain later modelling choices.

The brief does not prescribe particular filenames. The report content earns the marks, while the notebook, code, manifests, figures, and JSON records make that content traceable and reproducible.

## The application context that makes Q1 meaningful

The user is a textile Quality Assurance Inspector. The inspector submits a 256 × 256 grayscale fabric patch, reviews the model score, binary decision, and Grad-CAM overlay, and decides how to prioritise visual inspection. The human retains the release or hold decision.

This context is essential because a metric cannot be justified from class counts alone. The system's user and decision determine why missing a defect is more serious than requesting an unnecessary inspection.

Evidence:

- `README.md` states the user, prediction workflow, and limitations.
- `docs/approval-and-protocol.md` records the approved problem, user, decision, dataset, and supervised protocol.
- `reports/final_report.pdf`, page 1, states the application boundary and human responsibility.

## Requirement 1: describe the dataset

### What must be explained

A complete Q1 answer must identify where the data came from, whether it may be used, what the files represent, how many observations exist, and exactly what the prediction target means. Without these details, the reader cannot judge whether the experiment is lawful, reproducible, or scientifically meaningful.

### How this project answers it

- Dataset: RAW-FABRID V1.
- Authors: Pérez-Llorens, Albero-Albero, and Silvestre-Blanes.
- Publisher record: Mendeley Data, DOI `10.17632/db6g85xsyg.1`.
- Licence: CC BY 4.0.
- Modality: industrial line-scan, 8-bit grayscale fabric images.
- Full patch collection: 19,852 patches, each 256 × 256 pixels.
- Labels: 19,165 normal and 687 defective patches.
- Source images: 709 high-resolution images, each 1792 × 1024 pixels.
- Original high-resolution masks: 204.
- Publisher layout: 14,196 `train/good`, 4,969 `test/good`, and 687 `test/defect` patches.

The publisher's original layout is an anomaly-detection layout because its training folder contains no defective examples. This coursework needs supervised ANN and classical-model comparisons, so all explicitly labelled patches were pooled and reassigned before model development using source and publisher-origin grouping. Therefore, the reported results are for a new frozen supervised protocol and must not be presented as results on the publisher's original split.

### Target definition

The binary target is:

- `y = 1` when the corresponding mask contains at least one exact-white pixel with intensity 255;
- `y = 0` only when normal status is explicitly confirmed by the publisher data.

A normal patch remains normal even when another patch from the same parent image contains a defect. A missing mask is not, by itself, evidence that a patch is normal. Masks, source identifiers, and revealing paths are used for auditing but are not model inputs.

This precise target definition is mandatory because changing the mask threshold or assigning parent-level labels would change the task and the class distribution.

### Evidence files and why they matter

| Evidence | Role in satisfying Q1 |
|---|---|
| `scripts/download_data.py` | Records the dataset version, DOI, source, licence, download URL, archive size, and checksum. The full archive exceeds 500 MB, so the brief requires a reproducible download procedure and representative sample. |
| `data/sample/` and `data/sample-manifests/` | Provide authentic, testable examples without submitting the multi-gigabyte full archive. |
| `scripts/prepare_data.py` | Reads the publisher schemas, performs the metadata joins, validates the known V1 counts, and builds the supervised manifest. |
| `src/fabric_inspector/data/manifest.py` | Enforces image dimensions, label rules, mask alignment, hashes, group separation, and split immutability. |
| `data/manifests/split.json` and `split.csv` | Record every assessed observation and its frozen partition. |
| `data/manifests/data-audit.json` | Records the audited counts, source count, grouping field, target definition, mask policy, leakage results, and split hash. |
| `reports/final_report.pdf`, page 2 | Presents the concise, directly assessed dataset and target explanation. |

## Requirement 2: perform image-appropriate EDA

### Why this is mandatory

EDA must reveal properties that affect model design. For images, a class-count bar chart alone is not comprehensive. The analysis should examine label imbalance, annotation scale, acquisition variation, texture, source dependence, and representative images. The brief warns that descriptive plots without modelling consequences receive limited credit.

### Train-only rule

All model-driving EDA uses only the training partition. Looking at validation or test distributions to choose preprocessing, architecture, or thresholds would leak information from those partitions into development.

The training EDA reports:

- 13,384 training patches;
- 539 defective patches, or 4.027%;
- 478 source images;
- four independent publisher-origin components;
- 146 training source images containing at least one positive patch;
- defect area from one pixel, 0.0015%, to 22.37%;
- median defect area 4.72%, with quartiles 1.39% and 6.04%;
- patch mean intensity from 68.6 to 222.1; and
- grayscale standard-deviation contrast from 29.4 to 57.3.

The EDA also measures texture total variation and shows representative positive and normal training patches. These are acquisition and annotation observations, not invented semantic defect categories.

### Evidence files and why they matter

| Evidence | Role in satisfying Q1 |
|---|---|
| `scripts/eda.py` | Loads only rows whose split is `train`, computes image and mask statistics, generates figures, and writes exactly three modelling constraints. |
| `docs/eda-statistics.json` | Stores the numerical EDA results and frozen split hash in machine-readable form. |
| `reports/figures/eda-summary.png` | Shows class balance, positive patches per source, defect area, brightness, contrast, and texture variation. |
| `reports/figures/eda-training-examples.png` | Shows representative training images and annotation contours. |
| `notebooks/01_dataset_and_eda.ipynb` | Provides an executed, readable walkthrough of the manifest, split assertions, statistics, figures, and findings. |
| `reports/final_report.pdf`, page 3 | Presents the concise, directly assessed train-only EDA. |

## Requirement 3: explain false-positive and false-negative consequences

### Why this is mandatory

Classification errors do not have equal practical effects. The brief requires their relative consequences because these consequences should influence the metric and the final operating policy.

- False negative: a defective patch is predicted normal. This can allow a defect to pass routine inspection and may cause downstream material waste.
- False positive: a normal patch is predicted defective. This can trigger unnecessary manual inspection or an unnecessary hold.

The project treats false negatives as more serious, but it correctly describes these as coursework assumptions rather than measured factory costs. The prototype must not automatically release fabric or replace normal human inspection.

Evidence is stated on page 4 of `reports/final_report.pdf` and implemented consistently in `src/fabric_inspector/evaluation/metrics.py`.

## Requirement 4: define and justify the primary metric

### Why accuracy is unsuitable

Only 539 of 13,384 training patches are defective. An always-normal classifier would therefore obtain approximately 95.97% training accuracy while detecting no defects. Accuracy would reward the dominant class and conceal total failure on the class that matters operationally.

### Why defect-class F2 is primary

The primary metric is positive-class F2:

`F2 = 5TP / (5TP + FP + 4FN)`

F2 weights recall more heavily than precision while still penalising false alarms. That matches the stated decision context: a missed defect is more consequential than an additional inspection, but unlimited false alarms would still make the system unusable.

Precision, recall, F1, false-positive rate, confusion counts, average precision, and ROC-AUC are supporting metrics. They diagnose behaviour but do not replace the declared primary metric.

For score-producing models, the decision threshold is selected only on the full validation partition. `select_threshold` evaluates 101 fixed score quantiles plus lower and upper extremes, maximises validation F2, then breaks ties by precision and the higher threshold. The test set is not used for threshold selection.

Evidence:

- `src/fabric_inspector/evaluation/metrics.py` implements the F2 formula and validation-only threshold search.
- `reports/final_report.pdf`, page 4, gives the user- and imbalance-based justification.
- `docs/eda-modelling-findings.json` connects class imbalance to F2 and loss-weighting experiments.

## Requirement 5: report leakage checks and data-quality handling

### Why this is mandatory

Patches from the same high-resolution source or publisher origin can share weave, lighting, and acquisition characteristics. A random patch split could put highly related images in both training and test data, producing an unrealistically optimistic result. Leakage control is therefore part of the validity of every later model comparison.

### Frozen split and grouping

The project uses one immutable split for ANN, classical, and VLM evaluation:

| Partition | Publisher `roll_id` origin groups | Patches | Positive | Sources |
|---|---:|---:|---:|---:|
| Train | Rollo2A, Rollo4A, Rollo5A, Rollo7A | 13,384 | 539 | 478 |
| Validation | Rollo6A | 3,472 | 52 | 124 |
| Test | Rollo1A, Rollo3A | 2,996 | 96 | 107 |

`roll_id` must be described carefully. The publisher documentation identifies it as an anonymised fabric-origin field. The project keeps each value intact, but the repository does not prove that each value is exactly one physical roll, fabric type, or manufacturer.

The splitter considered 128 deterministic candidate group assignments and selected draw 32 using class availability and partition-balance criteria only. No model was trained or scored to select the split. The split SHA-256 is `af7daf13d0854ceeba004f1a06182bac778ac716021dec3d8fc40edf265aa76f`.

### Duplicate and near-duplicate checks

- Exact decoded-pixel hashes found zero duplicates.
- A 64-bit difference-hash screen examined 324,109 candidate pairs.
- 7,587 candidates reached full-resolution pixel comparison.
- Zero near matches were confirmed at mean absolute pixel difference no greater than 3 intensity levels, equivalent to 3/255 on a normalised scale.
- Source IDs, publisher-origin groups, exact hashes, and confirmed duplicate components are asserted not to cross partitions.

The audit is conservative but not perfect. Rotations, non-identical crops, and undocumented physical relationships can evade it. Reporting this limitation is part of technically honest leakage analysis.

### Data-quality handling

The implementation also checks that:

- every patch is 256 × 256, 8-bit grayscale;
- every positive patch has an aligned mask;
- missing masks are not silently interpreted as normal;
- normal labels have explicit publisher confirmation;
- mask-derived labels agree with the publisher defect areas; and
- raw mask bytes remain unchanged.

The audit found 471 patch masks with intermediate grayscale edge values. Counting only intensity 255 reproduces the publisher's 687 defect-area values. The source implements this exact-white policy; reproducing the count itself requires the full downloaded release.

Evidence:

- `src/fabric_inspector/data/manifest.py` implements validation, hashing, grouping, separation assertions, and immutable split creation.
- `scripts/prepare_data.py` performs explicit publisher metadata joins and count checks.
- `data/manifests/data-audit.json` records the frozen audit results.
- `docs/approval-and-protocol.md` explains the grouping semantics and protocol limitations.
- `reports/final_report.pdf`, page 4, gives the directly assessed summary.

## Requirement 6: end with three modelling constraints

### Why this is mandatory

The brief explicitly asks for three EDA findings that materially constrain modelling. A useful finding has the form `observation → consequence → modelling action`; a descriptive statistic alone is insufficient.

### Constraint 1: severe class imbalance

- Observation: 539 of 13,384 training patches are defective, approximately 4.03%.
- Consequence: accuracy can be high for a useless always-normal classifier, and unweighted training can neglect the positive class.
- Modelling action: use defect-class F2 as the primary metric, retain trivial references, and compare weighted and unweighted binary cross-entropy candidates. The train-derived positive weight is `(13,384 - 539) / 539 = 23.8312`.

Weighting is an experimental candidate, not a claim that every model must use it, and the class weight is not a measured business-cost ratio.

### Constraint 2: defects can be extremely small

- Observation: the smallest positive training annotation is one pixel, approximately 0.0015% of a 256 × 256 patch.
- Consequence: resizing, cropping, or blur can erase the only labelled evidence, while coarse feature maps can dilute it.
- Modelling action: preserve the 256 × 256 input geometry, avoid destructive input transformations, and explicitly examine the limitations of coarse Grad-CAM maps.

This does not mean that the CNN contains no internal downsampling. The selected MobileNetV3 reaches an 8 × 8 final feature map; that is a documented limitation rather than evidence that the input was resized.

### Constraint 3: few independent origins

- Observation: 13,384 training patches come from 478 source images but only four independent publisher-origin components.
- Consequence: patch count overstates the amount of independent acquisition diversity, so a high-capacity model may memorise origin-specific texture.
- Modelling action: restrict capacity, compare transfer learning, keep origin groups isolated, and use component-level rather than patch-independent uncertainty estimates.

The machine-readable authoritative version of these three findings is `docs/eda-modelling-findings.json`.

## End-to-end Q1 evidence flow

1. `scripts/download_data.py` records and verifies the publisher archive.
2. `scripts/prepare_data.py` joins publisher metadata and calls the manifest validation code.
3. `src/fabric_inspector/data/manifest.py` validates labels and masks, groups related images, audits duplicates, and freezes one split.
4. `data/manifests/` preserves the split and audit evidence.
5. `scripts/eda.py` reads only the training partition and generates statistics, figures, and exactly three constraints.
6. `notebooks/01_dataset_and_eda.ipynb` presents the executable walkthrough.
7. `reports/final_report.pdf`, pages 2–4, gives the concise answer assessed under Question 1.

## Technical accuracy corrections to the earlier draft

The earlier pasted explanation was substantially accurate, but it should be corrected as follows when studying or presenting it:

- Use paths from `AI-course-work-mandatory`, not the original `AI-course-work` worktree.
- The submitted report is `reports/final_report.pdf`; there is no required `reports/final_report.md` in this worktree.
- The training-example figure is `reports/figures/eda-training-examples.png`, not `compact-training-examples.png`.
- Do not cite the removed `data/README.md`; acquisition and provenance are covered by the root `README.md`, download script, manifest audit, and report.
- Refer to `roll_id` values as anonymised publisher-origin groups unless stronger physical-roll evidence is obtained.
- Say that input resolution and geometry are preserved and destructive input transformations are avoided. Do not claim that the CNN performs no internal downsampling.
- Say that weighted and unweighted loss were compared. Do not imply that the EDA proved class weighting must always be used.

## Viva self-check

You should be able to answer these questions without reading the code:

1. Why could the publisher's original `train/good` folder not directly train a supervised defect classifier?
2. Why is a missing mask insufficient evidence for a normal label?
3. Why are source and publisher-origin groups kept in only one partition?
4. What do the exact- and near-duplicate checks detect, and what can they miss?
5. Why would approximately 96% accuracy still represent a useless training-set classifier?
6. Why does F2 suit the stated inspection decision better than accuracy?
7. Why must threshold selection use validation scores rather than test scores?
8. How does each of the three EDA findings change a later modelling choice?
9. Why is `roll_id` not described as proven physical-roll identity?
10. Which Q1 claims require the full downloaded dataset to reproduce, and which can be checked from the submitted manifests?
