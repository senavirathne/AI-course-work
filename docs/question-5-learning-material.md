# Question 5 learning material: explainability and model insight

This guide explains how the submitted project answers Question 5 of the coursework brief, why each component is required, and how to interpret the evidence accurately. It is a study and viva aid; the assessed Q5 answer is on pages 18–22 of `reports/final_report.pdf`.

## What Question 5 requires

Question 5 has two connected parts.

### Q5.a: compare and select explainability approaches

The answer must:

1. discuss at least three relevant explainable-AI approaches;
2. explain how they work, what they explain, and their main strengths and weaknesses;
3. select a suitable technique for the trained ANN and intended user; and
4. justify the selection in relation to model architecture, data, latency, and user needs.

### Q5.b: apply and critically evaluate the selected approach

The answer must:

1. apply the selected explanation method to the trained model;
2. inspect at least three incorrectly predicted examples;
3. relate the explanations to the input, prediction, score, and known labels or annotations;
4. perform explanation sanity checks rather than relying only on visually attractive heatmaps; and
5. discuss whether each explanation is useful, misleading, incomplete, or unsupported.

The objective is not to prove that a heatmap is correct. The objective is to show that explanations are generated from the real trained model and to evaluate honestly what evidence they provide.

## How Questions 1–4 constrain Question 5

| Earlier evidence | Mandatory Q5 consequence |
|---|---|
| Defects may occupy extremely small parts of a 256 × 256 patch. | Check whether explanation maps spatially correspond to the publisher's defect markings and acknowledge coarse-map dilution. |
| The selected ANN is MobileNetV3-Small. | Choose a method compatible with convolutional feature maps and identify the exact target layer. |
| The task is binary patch classification, not segmentation. | Explain the defect-class decision without presenting the heatmap as a predicted boundary. |
| The output is one defect logit followed by a frozen threshold. | Target the defect logit consistently, including when the predicted class is defect-free. |
| The intended user is a QA inspector. | Keep the original patch visible and use the explanation as a visual inspection cue, not an automatic decision. |
| Q3 identified 28 false negatives and 42 false positives. | Analyse genuine mistakes rather than showing only successful or convenient examples. |

This continuity is mandatory because an explanation is meaningful only for a clearly identified model output, layer, input, and decision.

## Q5.a requirement 1: three explainability approaches

The report compares Grad-CAM, Integrated Gradients, and LIME.

| Method | Basic mechanism | Strengths for this project | Important limitations |
|---|---|---|---|
| Grad-CAM | Weights convolutional feature maps using gradients of a target output, combines them, applies ReLU, and upsamples the result. | Uses the selected CNN's spatial features; relatively fast; produces an intuitive spatial cue; integrates naturally into the web app. | Coarse and layer-dependent; shows only positive retained evidence after ReLU; not causal; not segmentation; normalisation can make weak and strong maps look similarly bright. |
| Integrated Gradients | Integrates input gradients along a path from a chosen baseline to the real input. | Produces pixel-level attribution and has useful completeness properties under its assumptions. | Requires many gradient evaluations; attribution depends on the baseline and integration settings; black or mean fabric may be unnatural; pixel maps can be noisy. |
| LIME | Perturbs image regions and fits a local interpretable surrogate around one prediction. | Model-agnostic and can explain any prediction interface. | Expensive; sensitive to superpixels, perturbations, sampling, and random seed; cutting periodic weave into superpixels can create off-distribution inputs; local surrogate fidelity must be measured. |

Comparing methods is mandatory because choosing Grad-CAM merely because it is familiar would not establish suitability. Each alternative becomes preferable under different constraints.

## Q5.a requirement 2: why Grad-CAM was selected

Grad-CAM is selected because:

- the deployed model contains convolutional spatial feature maps;
- the inspector needs a quick indication of where positive defect evidence is concentrated;
- it requires one forward and one backward computation rather than hundreds of perturbations;
- it can be returned with request-time predictions in the web application; and
- it produces a spatial overlay that can be viewed beside the unchanged original patch.

Integrated Gradients could be useful when input-level attribution and baseline sensitivity are studied carefully. LIME could be useful for a model without accessible gradients. For this CNN and interactive application, Grad-CAM offers the best practical architecture/user fit, not guaranteed faithfulness.

## How the Grad-CAM implementation works

For an input image and the scalar defect logit `y`, the implementation:

1. registers a forward hook on the chosen convolutional layer;
2. runs the model in evaluation mode and captures its feature maps `Aᵏ`;
3. computes gradients `∂y/∂Aᵏ`;
4. spatially averages each gradient map to obtain channel weight `αₖ`;
5. forms `ReLU(Σₖ αₖAᵏ)`;
6. bilinearly upsamples the map to 256 × 256; and
7. divides by its maximum when positive values exist.

In compact notation:

```text
αₖ = spatial mean of ∂y/∂Aᵏ
Grad-CAM = ReLU(Σₖ αₖAᵏ)
```

The target is always the **defect logit**, even for a patch predicted defect-free. This consistency makes maps comparable in meaning: bright regions represent positive contribution retained for the defect output, not whichever class happened to win.

The final MobileNet uses `network.features[-1]` as its CAM layer. Its feature map is spatially coarse relative to 256 × 256 input, so interpolation creates a smooth visual map but does not recover lost boundary detail. For the custom CNN, the last block's ReLU before its final pooling operation is used.

If the CAM contains no positive attribution, the implementation returns a genuine all-zero map. It does not divide by zero or fabricate a hotspot. Non-finite maps raise an error.

Evidence: `src/fabric_inspector/explainability/gradcam.py` and `src/fabric_inspector/models/neural.py`.

## What a Grad-CAM overlay means—and does not mean

A bright region means that, at the selected layer and for this input, positive Grad-CAM attribution for the defect logit is concentrated there.

It does **not** establish that:

- the bright region is an exact defect boundary;
- the highlighted pixels caused the prediction;
- the model discovered the physical manufacturing cause;
- a normal patch contains a hidden real defect;
- a dark region was irrelevant; or
- two separately normalised heatmaps have equal attribution magnitude.

These distinctions are mandatory. Grad-CAM is a post-hoc attribution method for a classifier, not a supervised segmentation output or causal explanation.

## Q5.b requirement 1: deterministic explanation sample

The explanation study uses the frozen `ann-search-018` model and held-out predictions. It selects up to nine cases from each confusion category:

| Category | Test population | Explanation sample |
|---|---:|---:|
| True positive | 68 | 9 |
| False negative | 28 | 9 |
| False positive | 42 | 9 |
| True negative | 2,858 | 9 |
| Total | 2,996 | 36 |

Within each category, selection prefers distinct source images and then uses deterministic filename order. The sample contains 18 genuine errors.

This rule matters because choosing only visually persuasive heatmaps after seeing the results would introduce selection bias. The 36-case explanation sample does not replace the full 2,996-patch predictive evaluation.

## Q5.b requirement 2: sanity check with controlled occlusion

Visual plausibility is not enough. A heatmap can look convincing even when the highlighted area is not important to the model output.

For every sampled case, the implementation selects equal 10% pixel areas from:

- the highest Grad-CAM values;
- the lowest Grad-CAM values; and
- a random seed-42 control.

Each selected area is replaced with the within-image channel mean. The measured response is:

```text
logit drop = original defect logit − perturbed defect logit
```

- A positive drop means the perturbation reduced defect evidence.
- A negative drop means the perturbation increased defect evidence.
- Strong local sensitivity requires a positive top-region drop that exceeds both the low-attribution and random controls.

All masks cover the same area, preventing area size from explaining the comparison. Under this strict rule, only **13 of 36** cases support stronger positive top-region sensitivity than both controls. This is limited evidence, not universal validation of Grad-CAM faithfulness.

Occlusion itself has limitations. Mean-filled blocks may disrupt the periodic weave and produce an unnatural image; correlated pixels and nonlinear interactions can make deletion effects difficult to interpret. Therefore the check supports or challenges local sensitivity, but it is not causal proof.

## Q5.b requirement 3: use the defect markings

For positive patches, the publisher provides a foreground defect mask. The study evaluates two correspondence measures.

### Pointing-game hit

The maximum-attribution pixel counts as a hit when it lies inside the defect mask. Only **2 of 18** positive annotated cases have a pointing hit.

### Attribution mass in the mask

The fraction of total CAM intensity inside the mask is compared with the fraction of image area occupied by the mask. A value above the uniform-area reference shows concentration inside the marked region.

This is more informative than visual inspection alone, but mask correspondence is not faithfulness. A map can overlap the annotation without the model depending on it, and it can be sensitive to a contextual area outside the exact mask.

For normal patches the publisher mask is empty. Pointing hit and attribution mass are therefore **undefined**, not zero. Reporting a zero would incorrectly treat the absence of a positive target as failed localisation.

## Q5.b requirement 4: analyse three genuine mistakes

The three retained cases are selected from the predeclared explanation sample: two false negatives and one false positive, so both available error types are represented.

### Error 1: tiny-defect false negative

`Rollo1A_0000005_D_07-05-2024_10-31-00.029_2_5.png`

| Field | Evidence |
|---|---:|
| Ground truth / prediction | defective / defect-free |
| Score / frozen cutoff | 0.790539 / 0.895234 |
| Marked defect area | 103 pixels, 0.157% |
| Pointing hit | No |
| CAM mass in mask | 0.00134% |
| Uniform-area reference | 0.15717% |
| Top / low / random logit drop | +1.591740 / +2.704260 / +0.498417 |

The marked defect is a tiny bottom-edge region, while the brightest CAM lies higher in the lower-right interior. CAM mass inside the mask is far below its area reference. Removing the top-CAM area reduces defect evidence, but removing the low-attribution control reduces it even more. Therefore this case fails the stronger-than-both-controls test.

The evidence is consistent with poor attention to tiny boundary detail, but it does not prove downsampling alone caused the error.

Panel: `reports/figures/xai-error-1.png`.

### Error 2: partly aligned false negative

`Rollo1A_0000018_D_07-05-2024_10-31-14.432_2_0.png`

| Field | Evidence |
|---|---:|
| Ground truth / prediction | defective / defect-free |
| Score / frozen cutoff | 0.204645 / 0.895234 |
| Marked defect area | 1,764 pixels, 2.692% |
| Pointing hit | No |
| CAM mass in mask | 21.3191% |
| Uniform-area reference | 2.6917% |
| Top / low / random logit drop | −0.427480 / −0.421962 / −0.641229 |

The CAM overlaps part of the narrow left-edge marking and concentrates more mass inside it than uniform area would predict, although its maximum remains outside. All three occlusions increase the defect logit, and the top drop is not stronger than the low-attribution control. Spatial correspondence therefore does not establish selective positive support for the decision.

Panel: `reports/figures/xai-error-2.png`.

### Error 3: false positive on a normal patch

`Rollo1A_0000002_D_07-05-2024_10-30-56.352_3_4.png`

| Field | Evidence |
|---|---:|
| Ground truth / prediction | defect-free / defective |
| Score / frozen cutoff | 0.925760 / 0.895234 |
| Publisher mask | Empty |
| Pointing/mass metrics | Undefined |
| Top / low / random logit drop | +2.779529 / +3.000914 / −0.592691 |

The CAM peaks near a short dark vertical texture feature. This suggests the model is sensitive to that feature, but it does not establish an unannotated real defect or an incorrect publisher label. Removing the low-attribution region reduces the defect logit more than removing the top region, so this case also fails the stronger-than-both-controls check.

Independent inspector annotation would be required before questioning the benchmark label.

Panel: `reports/figures/xai-error-3.png`.

## Why adverse explanation results are mandatory evidence

All three selected mistakes fail the stronger-than-both-controls perturbation check. Keeping those results is scientifically important because Q5 asks for critical evaluation, not promotional visualisation.

The correct overall conclusion is:

- Grad-CAM provides an intuitive, inexpensive spatial cue;
- some cases show useful local sensitivity;
- spatial correspondence with annotations is generally weak in this sample;
- attractive overlap does not guarantee faithfulness;
- the map can miss tiny defects or emphasise normal texture; and
- the inspector must consider the original image, score, threshold, and known model limitations together.

## Why explanations must not be used to retune after test access

The explanation study uses held-out cases. It describes the frozen model but does not change weights, threshold, architecture, or explanation settings afterward.

If the three errors were used to redesign the model and the same test set were scored again as if untouched, the test set would become development data. A future boundary-preserving model or alternative explanation validation must be evaluated under a new preregistered validation/test protocol.

## Technical-accuracy audit

The Q5 implementation and retained evidence are technically coherent, subject to these explicit limitations:

1. Grad-CAM targets the defect logit and the selected MobileNet feature layer consistently.
2. ReLU means the map retains positive evidence only; absence of brightness does not prove irrelevance.
3. Per-image maximum normalisation preserves within-image pattern but prevents magnitude comparison between images.
4. Bilinear upsampling improves display resolution, not the underlying spatial resolution of the feature map.
5. The deterministic 36-case sample avoids convenient cherry-picking but remains a sample, not complete explanation validation.
6. Only 13 of 36 cases pass the strict top-versus-both-controls sensitivity rule.
7. Only 2 of 18 annotated positive cases have a peak inside the defect mask.
8. Mask overlap measures correspondence, whereas controlled perturbation probes sensitivity; neither proves causality.
9. Mean-fill occlusion can produce unrealistic weave and is sensitive to the chosen area fraction and random control.
10. Empty masks make localisation metrics undefined; they do not justify assuming a false positive is an unlabelled defect.
11. The publisher annotation is benchmark ground truth but does not independently establish defect severity or manufacturing cause.
12. Three error panels support detailed qualitative analysis but cannot estimate population-wide explanation reliability.
13. Explanation findings are post-test descriptions and are not used for model retuning.

## Evidence map

| Q5 requirement | Primary repository evidence |
|---|---|
| Three methods and selection rationale | `reports/final_report.pdf`, page 18 |
| Grad-CAM implementation | `src/fabric_inspector/explainability/gradcam.py` |
| Exact CAM layer | `src/fabric_inspector/models/neural.py` |
| Deterministic sample and occlusion study | `scripts/xai_validate.py` |
| Full 36-case measured evidence | `reports/final-evaluation/xai/xai-validation.json` |
| Three genuine mistakes | `docs/three-xai-errors.json` |
| Visual error panels | `reports/figures/xai-error-1.png`, `xai-error-2.png`, `xai-error-3.png` |
| Executed walkthrough | `notebooks/03_serving_and_explainability.ipynb` |
| Live explanation endpoint | `app/main.py`, `src/fabric_inspector/serving/predictor.py` |
| Assessed answer | `reports/final_report.pdf`, pages 18–22 |

## How to reproduce the explanation evidence

With the complete dataset in its expected location and the environment installed:

```bash
PYTHONPATH=src python scripts/xai_validate.py
```

The frozen final-evaluation version is stored under `reports/final-evaluation/xai/`. Reproduction must respect the frozen protocol and should write to a fresh output location rather than silently modifying assessed evidence.

## Concise viva explanation

Question 5 requires me to compare explainability methods, justify one, apply it to the real model, and critically validate it on mistakes. I compared Grad-CAM, Integrated Gradients, and LIME and selected Grad-CAM because MobileNetV3 has convolutional feature maps and the inspector needs a fast spatial cue. The implementation targets the defect logit, weights the final feature maps by their gradients, applies ReLU, and upsamples the result. I evaluated a deterministic 36-case sample with publisher masks and equal-area top, low, and random occlusion. Only 13 cases passed the strict local-sensitivity test, and only 2 of 18 positive cases had the CAM peak inside the defect mask. The three analysed mistakes—two false negatives and one false positive—also failed the stronger-than-both-controls test. Therefore Grad-CAM is useful as a prompt for human inspection, but it is not a reliable defect boundary, causal explanation, or substitute for the original image and inspector judgement.
