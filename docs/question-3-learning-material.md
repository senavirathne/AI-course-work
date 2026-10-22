# Question 3 learning material: development, evaluation, and optimisation

This guide explains how the submitted project answers Question 3 of the coursework brief, why each element is required, and how to interpret the evidence accurately. It is a study and viva aid; the assessed Q3 answer is on pages 9–16 of `reports/final_report.pdf`.

## What Question 3 requires

Question 3 is worth 20 marks and has two equally weighted parts.

### Q3.a: development, baseline comparison, and error analysis — 10 marks

The answer must:

1. describe the initial neural model and how it was improved;
2. present and interpret learning curves, using evidence to decide whether performance is limited by underfitting, overfitting, or another cause;
3. present a classical baseline, its tuning search space, and the number of configurations tested;
4. compare a trivial/reference model, classical baseline, initial ANN, and final ANN using the primary metric, training time, and inference latency; and
5. analyse final-model errors by meaningful classes, subgroups, input properties, or data-quality factors.

### Q3.b: in-depth study of optimisation techniques — 10 marks

The answer must:

1. discuss strengths and weaknesses of relevant optimisation techniques and compare them using project evidence;
2. investigate at least three hyperparameters, including suitable choices such as learning rate, batch size, and network depth;
3. compare at least two optimisers empirically and explain their mechanics;
4. report at least two substantive approaches that did not work and diagnose why; and
5. maintain a complete, consistent experiment log.

The brief does not require particular filenames. The report contains the assessed argument; the files below make every important statement traceable.

## Experimental protocol: why it is mandatory

An apparent model improvement is not credible unless the comparison protocol prevents test-set tuning and records what changed.

The project uses one immutable, origin-grouped split. Candidate models, thresholds, and checkpoints are selected on validation data. The held-out test partition is used only for the frozen final comparison. The study records:

- 20 ANN screening configurations, each with at most four epochs and patience three;
- one longer refinement of the selected ANN candidate, still treated as a candidate;
- three selected-model seeds: 42, 123, and 2026;
- 24 main classical configurations from two feature sets, two kernels, three `C` values, and two class-weight choices;
- four retained classical pilot runs; and
- one frozen final evaluation record.

This structure is mandatory because Q3 asks for development and optimisation evidence, not merely one successful model. `reports/experiment-log.csv` contains 54 completed rows: 23 ANN, 28 SVM, two VLM, and one final-evaluation record. Crashes, parser checks, and interrupted engineering runs are not counted as scientific experiments.

Important limitation: validation contains only one publisher-origin component and 52 defective patches. Consequently, small validation differences are unstable. The guide reports observed results and avoids treating them as universal truths.

## Q3.a requirement 1: initial model and improvement path

### Initial ANN

`ann-search-000` is a 5,977-parameter scratch CNN:

```text
256×256 grayscale input
  → [Conv → BatchNorm → ReLU → MaxPool] × 3
  → global average pooling
  → dropout(0.2)
  → one binary logit
```

It uses AdamW, learning rate `0.001`, batch size `32`, weighted binary cross-entropy, flips, seed `42`, and a four-epoch budget. It is a sensible starting point because it tests whether a compact network can learn local weave disruption without excessive capacity.

### Improvement process

The study varies learning rate, batch size, depth, width, dropout, weighting, augmentation, optimiser, and transfer learning. The validation-selected final candidate, `ann-search-018`, is a genuinely ImageNet-pretrained MobileNetV3-Small with all layers fine-tuned. It uses AdamW at `0.0003`, batch size `32`, weighted loss, no augmentation, and seed `42`. Its validation-selected checkpoint is epoch 2.

The development claim is deliberately bounded: the whole initial-to-final change combines architecture, pretraining, optimiser settings, and augmentation choices. It demonstrates that the selected development path worked better, but it does not prove that pretraining alone caused the improvement.

Evidence: `experiments/ann-search-000.json`, `experiments/ann-search-018.json`, the architecture CSV files, `scripts/select_models.py`, and `notebooks/02_ann_and_optimisation.ipynb`.

## Q3.a requirement 2: learning curves and diagnosis

`reports/figures/initial-final-learning.png` presents the recorded train and validation loss trajectories.

| Model | Train BCE, first→last | Validation BCE, first→last | Selected epoch | Validation F2 | Validation AP |
|---|---:|---:|---:|---:|---:|
| Initial ANN | 1.2777→1.2564 | 1.1875→1.1716 | 1 of 4 | 0.0923 | 0.0188 |
| Final ANN | 0.9116→0.2142 | 0.7790→0.5912 | 2 of 4 | 0.4856 | 0.4991 |

The initial model's loss changes little and it produces 1,973 validation false positives. This supports limited early discrimination within the four-epoch budget. Plausible causes include inadequate convergence, cross-origin shift, insufficient representation, and dilution of tiny defects by global pooling. The evidence does not isolate those mechanisms.

The final model learns substantially stronger ranking, but later epochs do not necessarily improve validation F2. Retaining the validation-best checkpoint therefore matters.

Do not infer ordinary overfitting merely from the train/validation loss gap. Weighted BCE combines different positive/negative proportions in the two partitions, so the raw losses are not directly comparable. Also, four epochs are too short to establish eventual convergence. The accurate conclusion is that the initial network is limited under the tested budget and distribution, while the final candidate has better validation discrimination.

## Q3.a requirement 3: classical baseline and search space

The classical baseline uses train-fitted scaling with LBP or HOG+LBP features and a support-vector classifier. The declared main grid is:

| Factor | Values |
|---|---|
| Features | LBP (160 dimensions), HOG+LBP (8,260 dimensions) |
| Kernel | linear, RBF |
| `C` | 0.1, 1, 10 |
| Class weighting | none, balanced |

This gives `2 × 2 × 3 × 2 = 24` main configurations, plus four retained pilots. The selected classical run is `svm-search-042`: HOG+LBP, RBF kernel, `C=0.1`, and no class weighting.

This baseline is mandatory because a neural network should earn its complexity. Texture descriptors encode a credible prior for woven material, and the classical model is trained and selected with the same split discipline as the ANN.

## Q3.a requirement 4: frozen held-out comparison

All entries below use the same 2,996 held-out patches and validation-frozen decisions.

| Model | F2 | Precision | Recall | TP/FP/FN | Training seconds | Mean end-to-end latency |
|---|---:|---:|---:|---:|---:|---:|
| Always normal | 0.0000 | 0.0000 | 0.0000 | 0/0/96 | 0.0 | not meaningful |
| Always defective | 0.1420 | 0.0320 | 1.0000 | 96/2900/0 | 0.0 | not meaningful |
| Classical SVM | 0.0623 | 0.2941 | 0.0521 | 5/12/91 | 169.3 | 36.91 ms |
| Initial ANN | 0.0000 | 0.0000 | 0.0000 | 0/0/96 | 841.2 | 6.46 ms |
| Final ANN | 0.6883 | 0.6182 | 0.7083 | 68/42/28 | 625.3 | 7.44 ms |

The always-defective reference is essential because F2 rewards recall: a model can obtain F2 `0.1420` by flagging every patch. The final ANN clearly exceeds both trivial references and the learned classical baseline. Accuracy alone would conceal failure because 96.80% of test patches are normal.

Supporting ranking results are AP `0.7185` and ROC-AUC `0.8989` for the final ANN, compared with AP `0.1014` and ROC-AUC `0.6127` for the SVM. These are supporting metrics; F2 remains the predeclared primary metric.

Timing conditions are part of the result: CPU-only Linux, one inference thread, batch size one, and three warmups. End-to-end latency includes file decode, preprocessing, and prediction but excludes process and network startup. Training time is elapsed CPU time, not a hardware-independent measure of computational complexity.

The nominal group-bootstrap F2 interval for the final ANN is `[0.6810, 0.6977]`, but test has only two independent origin components. The interval therefore has very few distinct resamples and must not be presented as reliable coverage of future production fabric.

The workspace-reset record discloses that held-out inference was physically recomputed after the original result files were lost. The frozen model, threshold, split, and protocol were unchanged. This is transparent recovery evidence, not an independent test replication.

## Q3.a requirement 5: final-model error analysis

### By held-out origin

| Publisher origin ID | Patches | Defects | F2 | Recall | FP/FN |
|---|---:|---:|---:|---:|---:|
| Rollo1A | 2,436 | 53 | 0.6810 | 0.7170 | 29/15 |
| Rollo3A | 560 | 43 | 0.6977 | 0.6977 | 13/13 |

The results are similar, but two groups are insufficient for a general domain-robustness conclusion. Publisher IDs must not be reinterpreted as verified manufacturers or physical rolls.

### By annotated defect area

| Defect-mask share | Positive patches | Missed | Recall |
|---|---:|---:|---:|
| ≤1% | 19 | 12 | 0.3684 |
| 1–5% | 40 | 11 | 0.7250 |
| >5% | 37 | 5 | 0.8649 |

The strongest pattern is poor recall for tiny marked regions: 12 of 19 are missed. This supports a tiny-defect limitation and connects to the spatial reduction/global-pooling design. It does not prove size is the sole cause because texture, contrast, and origin can be confounded. Mask area is annotation coverage, not verified defect severity.

The recommendation analysis places 2,858 patches in routine inspection, 116 in manual review, and 22 in priority hold. Review plus hold is 4.61% of the test set, yet 28 defects remain in routine inspection. These bands use coursework cost assumptions and uncalibrated scores; they do not certify safe automatic release. Error analysis is post-test description only and is not used to retune the model.

Evidence: `reports/error-analysis.json` and `reports/final-evaluation/comparison.json`.

## Q3.b requirement 1: optimisation techniques and trade-offs

The project evaluates techniques rather than listing them abstractly:

- **Class weighting** increases the influence of rare defects, but can create excessive false positives and does not calibrate scores.
- **Early stopping/checkpoint selection** prevents automatically using a worse last epoch, but a noisy one-origin validation set can select an unstable epoch.
- **AdamW** adapts updates using first and second moments and decouples weight decay. It is convenient for different gradient scales but adds optimizer state and may converge to a different solution from SGD.
- **Momentum SGD** accumulates a velocity from past gradients. It is simpler but can be more sensitive to learning rate and may need longer training.
- **Transfer learning** provides reusable visual features when independent data diversity is limited, but ImageNet colour features are not perfectly matched to grayscale fabric.
- **Augmentation** can add invariance, but only label-preserving transforms are valid and the selected final candidate did not benefit enough for augmentation to be retained.

Every strength is paired with a limitation because Q3.b asks for in-depth analysis, not a catalogue of techniques.

## Q3.b requirement 2: three hyperparameter studies

`reports/figures/hyperparameter-study.png` visualises the matched screens.

### Learning rate

| AdamW learning rate | Validation F2 |
|---:|---:|
| 0.0001 | 0.0804 |
| 0.0003 | 0.0941 |
| 0.001 | 0.0923 |
| 0.003 | 0.0881 |

`0.0003` is strongest in this four-epoch screen. A lower rate may need longer training; a higher rate can overshoot or fluctuate. The result is budget-specific, not a universal optimum.

### Batch size

| Batch size | Validation F2 |
|---:|---:|
| 16 | 0.0895 |
| 32 | 0.0923 |
| 64 | 0.0939 |

Batch 64 is marginally strongest, but the difference is small relative to the validation instability. Fixed epochs do not mean fixed optimisation work: batch 16 performs roughly twice as many parameter updates as batch 32. Therefore this study compares practical epoch-budget configurations, not gradient noise in isolation.

### Network depth

| Convolutional blocks | Validation F2 |
|---:|---:|
| 2 | 0.0793 |
| 3 | 0.0923 |
| 4 | 0.1799 |

Four blocks are strongest in this scratch-CNN screen. More depth changes parameter count, receptive field, and downsampling simultaneously. Extra pooling may also erase tiny signals or lower Grad-CAM resolution, so the study does not prove that still deeper models would be better.

## Q3.b requirement 3: two optimisers

| Optimiser | Learning rates tested | Best observed validation F2 |
|---|---|---:|
| AdamW | 0.0001, 0.0003, 0.001, 0.003 | 0.0941 |
| Momentum SGD | 0.001, 0.01, 0.03, 0.1 | 0.1274 |

Both use weight decay `1e-4`; SGD uses momentum `0.9` without Nesterov, and AdamW uses betas `0.9/0.999` and epsilon `1e-8`. No learning-rate scheduler is used. Architecture, loss, augmentation, batch size, seed, and maximum epochs are held constant.

The correct inference is that the best tested SGD configuration outperformed the best tested AdamW configuration under this short budget. It is not a pure optimizer-only causal comparison because the learning-rate grids differ, and it does not establish eventual SGD superiority.

## Seed stability

The final candidate is repeated using seeds 42, 123, and 2026.

| Seed | Validation F2 |
|---:|---:|
| 42 | 0.4856 |
| 123 | 0.3704 |
| 2026 | 0.4527 |

Mean validation F2 is `0.4362`, sample SD is `0.0594`, and the range is `0.1152`. This is material variability and reinforces the small-validation-set limitation. The three frozen test F2 values average `0.7013` with sample SD `0.0412`. Seed 42 remains the serving model; test ranking is not used to choose a seed.

## Q3.b requirement 4: failed approaches

### Substantive failure 1: LBP-only descriptor

`svm-search-024` tests 160-dimensional spatial LBP against the otherwise matched HOG+LBP control `svm-search-036`.

| Run | Validation F2 | AP | TP/FP/FN |
|---|---:|---:|---:|
| LBP only | 0.0740 | 0.0148 | 38/2323/14 |
| HOG+LBP control | 0.1252 | 0.0331 | 20/571/32 |

LBP-only obtains recall by generating 2,323 false positives. Adding oriented-gradient information improves AP and F2 and substantially reduces false positives, though recall falls. This supports retaining boundary/orientation information; it does not prove every possible LBP variant must fail.

### Substantive failure 2: short-budget scratch CNN

`ann-search-000` is compared with `svm-search-036`.

| Run | Validation F2 | AP | TP/FP/FN |
|---|---:|---:|---:|
| Scratch CNN | 0.0923 | 0.0188 | 41/1973/11 |
| HOG+LBP SVM | 0.1252 | 0.0331 | 20/571/32 |

The scratch CNN is not selective under the four-epoch budget. Possible causes include inadequate convergence, limited origin diversity, and global pooling of tiny regions, but the whole-method comparison cannot isolate one causal factor.

Two additional measured failures strengthen the record: AdamW learning rates `0.0001` and `0.003` both underperform the `0.001` control in their matched short-budget runs. They are supplemental tuning failures, not substitutes for the two substantive approach failures.

Only real completed experiments count. Syntax errors, runtime interruptions, or deliberately broken configurations would not satisfy this requirement. Full hypotheses, controls, diagnoses, and lessons are stored in `docs/failed-approaches.json`.

## Q3.b requirement 5: complete experiment record

The experiment log is mandatory because it prevents selective reporting. Each row records run ID, family, status, timestamps, seed, split/code hashes, hypothesis, configuration, completed and selected epochs, validation metrics, confusion counts, elapsed training time, checkpoint/hash, outcome, lesson, and evaluation partition.

`reports/experiment-log.csv` contains successful and unsuccessful completed runs. `docs/training-method.json` records optimiser defaults and reproducibility settings. `artifacts/seed-variability.json` records repeated-seed evidence. Together they let an assessor connect every table entry to an executed configuration.

## Technical-accuracy audit

The Q3 evidence is technically coherent and satisfies the requested components, subject to these explicit limitations:

1. The final ANN is strongly better on the frozen held-out F2, but only two independent test origins limit external validity.
2. Validation selection is unstable because it uses one origin and 52 positives; seed variation is material.
3. Initial-versus-final is a development comparison, not causal proof that pretraining alone produced the gain.
4. Optimiser/LR and depth studies are bounded practical comparisons; they do not isolate every mechanism.
5. Batch-size comparisons use equal epochs rather than equal optimizer updates.
6. Loss gaps under weighted BCE do not independently prove overfitting.
7. Latencies are valid only under the stated CPU protocol; elapsed training time is not a portable compute measure.
8. The bootstrap interval is descriptive because two groups give very coarse resampling.
9. Recommendation bands use assumed costs and uncalibrated scores; they are not operational safety validation.
10. The disclosed second physical test computation followed a workspace reset and unchanged frozen protocol; it is a recovery exception, not repeated model selection.

These qualifications strengthen the answer because they keep conclusions aligned with what the experiment can actually demonstrate.

## Evidence map

| Q3 requirement | Primary repository evidence |
|---|---|
| Initial and improved ANN | `experiments/ann-search-000.json`, `experiments/ann-search-018.json`, architecture CSVs |
| Learning curves | `reports/figures/initial-final-learning.png`, experiment JSON histories |
| Classical baseline and grid | `reports/experiment-log.csv`, `scripts/train.py`, selected SVM artifact metadata |
| Frozen comparison and timings | `reports/final-evaluation/comparison.json` |
| Error analysis | `reports/error-analysis.json` |
| Three hyperparameters | `reports/figures/hyperparameter-study.png`, matched ANN runs |
| Optimiser mechanics/configuration | `docs/training-method.json`, ANN experiment records |
| Seed variability | `artifacts/seed-variability.json` |
| Failed approaches | `docs/failed-approaches.json` |
| Executed walkthrough | `notebooks/02_ann_and_optimisation.ipynb` |
| Assessed answer | `reports/final_report.pdf`, pages 9–16 |

## Concise viva explanation

Question 3 requires evidence of a disciplined development process. I started with a small scratch CNN, used validation-only experiments to study learning rate, batch size, depth, optimiser, and other design choices, and retained a pretrained MobileNetV3-Small candidate. I compared it on the frozen test set with two trivial references, a tuned HOG+LBP SVM, and the initial ANN. The final ANN achieved F2 `0.6883`, but it still missed 28 of 96 defects and recall fell to `0.3684` for defects occupying at most 1% of a patch. I therefore report it as the strongest tested model, not as production-safe. Every completed experiment, including genuine failures, is retained in the log, and the small number of independent origins and seed variation limit how broadly the result can be generalised.
