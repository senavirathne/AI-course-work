# Question 6 learning material: generative AI and foundation-model evaluation

This guide explains how the submitted project answers Question 6 of the coursework brief, why each component is required, and how to interpret the evidence accurately. It is a study and viva aid; the assessed Q6 answer is on pages 23–24 of `reports/final_report.pdf`.

## What Question 6 requires

Question 6 has two connected parts.

### Q6.a: apply and evaluate a relevant generative/foundation model

The answer must:

1. identify a genuine foundation or generative model suitable for the project;
2. define a meaningful task connected to the approved problem;
3. document the exact model, prompt, inference configuration, and output-handling procedure;
4. evaluate real model outputs rather than simulate them;
5. use a fair, leakage-controlled protocol; and
6. report weak results and generation failures as well as successes.

### Q6.b: compare, assess deployment, and propose a useful extension

The answer must:

1. compare the generative approach numerically and conceptually with the trained model;
2. discuss performance, latency, cost, privacy, reproducibility, and operational suitability;
3. explain when each approach would or would not be appropriate; and
4. propose a realistic extension that uses generative AI without inventing unsupported capability.

The coursework needs evidence of actual generative-model use and critical evaluation. Merely asking ChatGPT to write the report, describing a possible VLM, or presenting hand-written example outputs would not answer Q6.a.

## How Questions 1–5 constrain Question 6

| Earlier evidence | Mandatory Q6 consequence |
|---|---|
| The target is binary patch-level defect detection. | Give the VLM the same 256 × 256 patch task and map output to `defective` or `defect_free`. |
| Defects are rare and false negatives matter. | Use the same defect-class F2 and report precision, recall, false positives, and false negatives—not accuracy alone. |
| Model/threshold selection must use validation only. | Select the prompt on validation data and freeze it before test evaluation. |
| The final ANN was evaluated on all 2,996 test patches. | Evaluate the selected VLM prompt on the same complete held-out population. |
| Tiny defects can occupy very little image area. | Preserve 256 × 256 pixels and disclose the VLM's severe visual-token compression. |
| The application is local decision support. | Compare local latency, dependency, privacy, and failure behavior rather than assuming a hosted API. |
| Q5 distinguishes explanations from generated text. | Do not call a generated label or fluent sentence a faithful explanation. |

These links are mandatory for a fair comparison. A VLM tested on easier examples, different labels, or a different partition would not be comparable with the ANN.

## Q6.a requirement 1: select a genuine suitable model

The assessed model is:

| Field | Frozen value |
|---|---|
| Model | `HuggingFaceTB/SmolVLM-256M-Instruct` |
| Immutable revision | `7e3e67edbbed1bf9888184d9df282b700a323964` |
| Licence | Apache 2.0 |
| Library | Transformers 4.57.1 |
| Device | CPU, one thread |
| Data type | bfloat16 |
| Sampling | Greedy, `do_sample=False` |
| Maximum generated tokens | 24 |
| Batch size | 1 |
| Quantisation | None |
| Retry policy | One retry after inference/parsing failure |

SmolVLM is image-capable and instruction-following, so direct patch classification is a genuine use of a vision-language foundation model. Its approximately 256-million-parameter size makes local CPU evaluation feasible.

Suitability is not the same as expected superiority. The model is pretrained for broad vision-language behavior rather than fine fabric inspection. At this configuration, a 256 × 256 image is represented by only 16 matching visual tokens. The underlying pixels are preserved without application-side tiling or upscaling, but the internal visual compression can lose tiny defect evidence.

A larger VLM might represent subtle features better, but it would require more memory, download size, inference time, and possibly different hardware. Testing one would be a new experiment, not evidence already provided by this submission.

## Q6.a requirement 2: define a genuine task

The VLM performs direct zero-shot binary classification of every patch. It receives an image and an instruction and must output one label:

```text
defective
```

or

```text
defect_free
```

This is meaningful because it addresses the same inspection target as the trained ANN without task-specific weight fitting. It is a comparator, not a replacement for the mandatory trained neural model.

The task is deliberately narrow. Asking for free-form defect descriptions would make objective comparison with binary ground truth harder and would add unsupported semantic claims about severity or manufacturing cause.

## Q6.a requirement 3: prevent label and metadata leakage

The model's inference interface accepts only image pixels and the selected prompt. It does not receive:

- the ground-truth label;
- defect masks;
- patch filename or filesystem path;
- publisher-origin or source-group IDs;
- ANN scores or predictions; or
- test-derived hints.

PNG metadata is removed by converting the decoded image to a fresh RGB pixel image. Labels are used only after generation for scoring.

This is mandatory because many filenames and paths reveal dataset structure. A high score obtained from leaked labels or revealing metadata would not measure visual defect recognition.

## Q6.a requirement 4: compare prompt variants on validation data

Two prompts are evaluated on all 3,472 validation patches.

### Prompt 0

> Inspect this fabric patch for a visible manufacturing defect. Reply with only one label: defective or defect_free.

| F2 | Precision | Recall | Unresolved |
|---:|---:|---:|---:|
| 0.0707 | 0.0150 | 1.0000 | 74 (2.13%) |

### Prompt 1

> Classify this fabric patch. Look for a local disruption, hole, stain or anomalous yarn rather than ordinary repeating weave or lighting. Reply with only one label: defective or defect_free.

| F2 | Precision | Recall | Unresolved |
|---:|---:|---:|---:|
| 0.0707 | 0.0150 | 1.0000 | 23 (0.66%) |

The frozen selection rule is:

1. higher validation F2;
2. then lower unresolved-output rate; and
3. then smaller prompt index.

Both prompts have the same F2 and classify zero normal validation patches correctly. Prompt 1 is selected only because it has fewer unresolved outputs. Its extra wording improves format reliability, not measured discrimination.

This distinction is mandatory. Recall of 1.0 sounds impressive in isolation, but it results from an almost universal defective response. Prompt tuning is not allowed to use test results.

## Q6.a requirement 5: use a strict output parser

Generated text must be converted to a machine-evaluable decision without silently interpreting arbitrary prose.

The parser accepts:

- one exact `defective` or `defect_free` label, including the declared hyphen/space normalisations; or
- one canonical sentence such as “The fabric patch is defective.”

It rejects additional reasoning, multiple decisions, ambiguity, and unexpected formats. After the fixed retry, an unresolved result is retained as prediction `-1` and counted as an error for the example's true class.

The complete record stores raw response, attempt error where applicable, input/output token counts, resolved model revision, and latency. The system therefore does not conceal malformed generations or repair them using the true label.

One metric detail needs careful explanation: under the declared failure policy, an unresolved normal example contributes to the false-positive error count, while an unresolved defective example contributes to the false-negative count. Thus the confusion totals retain all examples even though unresolved is not literally a positive or negative generated label. The separate unresolved count preserves that distinction.

## Q6.a requirement 6: frozen held-out VLM result

The selected prompt is evaluated on the same 2,996 held-out patches as the ANN.

| Metric | VLM result |
|---|---:|
| F2 | 0.1391 |
| Precision | 0.0314 |
| Recall | 0.9792 |
| TP / FP / FN / TN | 94 / 2,900 / 2 / 0 |
| Unresolved outputs | 48 (1.60%) |
| Mean end-to-end latency | 764.18 ms |
| P95 end-to-end latency | 859.07 ms |

The model never returns a correct defect-free test decision: it produces either a defective decision or an unresolved output for normal cases. Of the 48 unresolved cases, the metric policy still counts each as wrong for its true class.

The VLM's F2 `0.1391` is slightly below the always-defective reference F2 `0.1420`. Its very high recall is therefore not useful discrimination; it creates an unmanageable false-positive burden.

The model produces labels rather than a continuous defect ranking. Average precision, ROC-AUC, calibration, and a probability threshold are therefore not invented for the VLM.

## Real evidence and reproducibility

The evaluation is not simulated:

- `src/fabric_inspector/genai/vlm.py` loads the real model and processor;
- model weights are resolved at an immutable publisher revision;
- `experiments/vlm-validation-0.json` and `vlm-validation-1.json` contain every validation example;
- `artifacts/vlm-selection.json` records the selected prompt and configuration;
- `reports/final-evaluation/vlm.json` contains all 2,996 held-out examples and attempts; and
- `notebooks/04_real_vlm_and_final_protocol.ipynb` contains the executed walkthrough.

The model weights are not presented as repository-authored artifacts; full reproduction requires downloading the pinned pretrained revision. The immutable revision improves repeatability, but future library/platform differences can still affect execution.

## Q6.b requirement 1: numerical comparison with the trained ANN

| Evidence | Trained ANN | Generative VLM |
|---|---:|---:|
| Test population | 2,996 | 2,996 |
| F2 | 0.6883 | 0.1391 |
| Precision | 0.6182 | 0.0314 |
| Recall | 0.7083 | 0.9792 |
| False positives | 42 | 2,900 under the declared unresolved-error policy |
| False negatives | 28 | 2 under the declared unresolved-error policy |
| True negatives | 2,858 | 0 |
| Mean end-to-end latency | 7.44 ms | 764.18 ms |
| Unresolved output | 0 | 48 |

The VLM is approximately 103 times slower by the recorded mean latencies. Both measurements use CPU, one thread, batch size one, and exclude one-time model loading. ANN timing includes decode, preprocessing, and prediction. VLM timing includes decode, processor work, generation, parsing, and retries. The workloads are different, so the ratio describes this deployment experiment rather than an architecture-independent speed law.

The final ANN is clearly more suitable for direct classification: it has much higher F2 and precision, far fewer false positives, lower latency, and a deterministic numerical interface. The VLM's higher recall is achieved by effectively flagging everything and should not be treated as superior safety.

## Q6.b requirement 2: conceptual comparison

| Consideration | Trained ANN | VLM |
|---|---|---|
| Task learning | Fine-tuned using labelled fabric data | No task-specific weight fitting; controlled through prompts |
| Output | Continuous uncalibrated score and deterministic class | Generated label with possible formatting failure |
| Dependencies | Small fixed inference stack and 5.9 MiB checkpoint | Larger pretrained model/processor download and generation stack |
| Adaptability | New task usually requires data and retraining | Prompt can quickly test adjacent language/image tasks |
| Fine-detail risk | Downsampling can miss tiny defects | Sixteen visual tokens create even stronger compression risk |
| Explanation | Measured Grad-CAM with disclosed limits | Fluent output is not automatically faithful explanation |
| Reproducibility | Local checkpoint and frozen threshold | Pinned revision and greedy decoding help, but model/library dependency remains larger |
| Failure mode | Numerical score always produced for valid input | Can produce unresolved or non-compliant text |

Neither family is universally superior. The evidence establishes that this ANN is better for this direct benchmark under the tested conditions.

## Q6.b requirement 3: cost and latency

The local VLM uses no paid inference API, so its recorded marginal API charge is zero. That does not make it cost-free. Relevant costs include:

- hardware and memory requirements;
- model download and storage;
- electricity;
- CPU time;
- dependency maintenance; and
- engineering effort for prompting, parsing, retries, and monitoring.

The study does not invent a monetary total because electricity price, hardware amortisation, and deployment volume were not measured. The measured latency is the defensible quantitative cost evidence.

A larger VLM could improve capability but would usually increase these burdens. That trade-off must be experimentally measured rather than assumed.

## Q6.b requirement 4: privacy, security, and provenance

Both assessed models run locally, so uploaded images stay within the local process during inference. The VLM adapter additionally strips image metadata and does not receive labels, paths, masks, or ANN results.

A hosted VLM would introduce additional issues:

- sending factory imagery to a third party;
- data retention and model-training policies;
- service availability and network latency;
- authentication and access control;
- changing prices and model versions; and
- weaker reproduction if the provider changes the model.

Local execution reduces external disclosure but does not by itself solve access control, encryption, logging, device security, or retention policy.

The pretrained model's data provenance also cannot prove that RAW-FABRID or related images were absent from pretraining. No benchmark-contamination claim is made.

## Q6.b requirement 5: deployment recommendation

For direct defect classification, deploy the trained ANN rather than the tested VLM because the ANN has substantially better F2, precision, latency, and output reliability.

The recommendation remains conditional:

- retain human inspection;
- monitor performance on new fabric origins;
- do not interpret the ANN score as calibrated confidence;
- do not use the VLM's near-universal defective decision as a safe inspection policy; and
- validate any new model, prompt, or task on origin-held-out data before deployment.

Architecture prestige is not a deployment criterion. The decision follows measured user costs, error types, latency, privacy, and reproducibility.

## Q6.b requirement 6: realistic generative-AI extension

A suitable extension is an inspection-note assistant. It would receive only verified structured fields, such as:

- inspector-confirmed defect status;
- patch identifier; and
- inspector-confirmed observed location.

It would produce a short structured note while being required to:

- quote the verified fields accurately;
- avoid inventing defect severity or root cause;
- identify missing information rather than guessing; and
- request human review before the note becomes part of a quality record.

This is a better generative use case because language generation adds value to documentation while the deterministic ANN and inspector retain the safety-critical classification decision.

A future study should evaluate factual-field accuracy, unsupported-claim rate, editing time, and usefulness on new origin-held-out cases. The extension is proposed, not claimed as implemented.

## Distinguishing assessed VLM use from AI development assistance

The assessed SmolVLM experiment is separate from ChatGPT/Codex assistance used during project development. The repository records both transparently:

- Q6 evidence concerns image-and-prompt inference by the pinned local SmolVLM;
- `docs/ai-use-declaration.md` records assistance with planning, implementation, analysis, documentation, and verification; and
- the student remains responsible for reviewing, understanding, and defending all submitted work.

This distinction is mandatory because productivity assistance is not a substitute for the genuine evaluated foundation-model experiment.

## Technical-accuracy audit

The Q6 implementation and evidence are technically coherent, subject to these explicit limitations:

1. Both prompts are evaluated on the complete validation population and the selected prompt is frozen before test evaluation.
2. Prompt 1 wins only on lower failure rate; neither prompt correctly classifies a normal validation patch.
3. The test VLM F2 is slightly worse than the always-defective trivial reference, so high recall is not evidence of useful discrimination.
4. Unresolved outputs are retained and counted as class-dependent errors; the separate count must accompany confusion metrics.
5. The VLM provides no continuous score, so AP, ROC-AUC, calibration, and probabilities are unavailable.
6. Sixteen visual tokens are a major limitation for subtle or tiny defects, even though the input pixels are not application-side resized or tiled.
7. Greedy decoding and a pinned revision improve reproducibility but do not eliminate hardware/library dependence.
8. Local execution has zero paid API charge, not zero compute or ownership cost.
9. Timing excludes one-time loading and is specific to CPU, one thread, batch one, and this dependency stack.
10. Unknown pretraining provenance prevents a guarantee of zero benchmark contamination.
11. Generated labels or prose are not faithful causal explanations.
12. A larger VLM and inspection-note assistant are proposed future studies, not completed results.
13. Neither model is validated for unseen production populations or autonomous release decisions.

## Evidence map

| Q6 requirement | Primary repository evidence |
|---|---|
| Model adapter, prompts, parser, retry policy | `src/fabric_inspector/genai/vlm.py` |
| Complete validation runs | `experiments/vlm-validation-0.json`, `experiments/vlm-validation-1.json` |
| Validation-only prompt selection | `scripts/vlm_validate.py`, `artifacts/vlm-selection.json` |
| Complete frozen test evaluation | `reports/final-evaluation/vlm.json` |
| ANN/VLM shared comparison | `reports/final-evaluation/comparison.json` |
| Frozen protocol | `artifacts/final-evaluation-manifest.json` |
| Executed walkthrough | `notebooks/04_real_vlm_and_final_protocol.ipynb` |
| AI-assistance declaration | `docs/ai-use-declaration.md`, report appendix |
| Assessed answer | `reports/final_report.pdf`, pages 23–24 |

## How to reproduce the validation experiment

After downloading the complete dataset and installing the VLM dependencies:

```bash
PYTHONPATH=src python scripts/vlm_validate.py --device cpu
```

The command downloads/loads the pinned model revision and evaluates both prompts on validation data. Existing result files are intentionally protected from overwrite. Full held-out evaluation belongs to the frozen final-evaluation workflow and must not be rerun casually for model or prompt selection.

## Concise viva explanation

Question 6 requires genuine and critically evaluated foundation-model use. I used the pinned local SmolVLM-256M-Instruct as a zero-shot binary classifier. I evaluated two exact prompts on all 3,472 validation patches and selected prompt 1 only because both prompts tied on F2 while it had fewer unresolved outputs. The frozen prompt was then evaluated on the same 2,996 test patches as the trained ANN. SmolVLM obtained F2 `0.1391`, precision `0.0314`, recall `0.9792`, 2,900 false-positive errors, 48 unresolved outputs, and mean latency `764.18 ms`. The ANN obtained F2 `0.6883` at `7.44 ms`. Therefore the VLM's high recall comes from effectively flagging everything and does not make it suitable for direct deployment. The ANN is the better classifier, while a safer future generative use is drafting inspection notes from inspector-verified fields with unsupported-claim checks and human approval.
