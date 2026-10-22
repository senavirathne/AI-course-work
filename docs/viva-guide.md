# Individual viva preparation

Read the report, run the app and inspect the actual code. These notes do not demonstrate your understanding for you.

1. Explain the specific user: textile Quality Assurance Inspector. A patch score changes inspection priority; the inspector makes the release/hold decision.
2. Explain why publisher train/good alone is unsuitable for supervised binary training. We pool labelled patches, then freeze source/duplicate/publisher-origin groups before any fit. This is a new supervised benchmark, not the publisher anomaly split.
3. State the full counts: 19,852 patches, 687 positive; train 13,384/539, validation 3,472/52, test 2,996/96. Seven anonymised origin IDs do not establish seven independently known physical rolls/manufacturers.
4. Explain exact-white masks: 471 masks have intermediate gray edges; intensity 255 reproduces all CSV areas. Do not copy parent labels or assume absent masks mean normal.
5. Derive F2 = 5TP/(5TP+FP+4FN), explain recall emphasis and why accuracy alone is weak. Validation cutoffs are frozen. AP needs continuous scores; the label-only VLM has no invented AP/AUC/probability.
6. Describe the initial 5,977-parameter CNN: Conv3x3/BN/ReLU/Pool blocks at widths 8/16/32; GAP/dropout 0.2/linear logit. Explain local texture assumptions, tiny-defect dilution, normalisation, train-derived positive weight 23.8312 and BCEWithLogitsLoss.
7. Describe final ann-search-018: {'name': 'mobilenet'}, 1,518,881 parameters. Inspect its actual architecture CSV and checkpoint metadata; do not repeat an obsolete width-16 parameter count.
8. Trace one controlled learning-rate, batch-size and actual-depth comparison. Fixed epochs are not fixed update counts. Compare AdamW and momentum SGD at their own tested rates; four epochs do not establish eventual optimiser superiority.
9. Defend the classical budget: 24 main LBP/HOG+LBP, linear/RBF, C .1/1/10, class weights None/balanced configurations; four historical pilots. Explain train-only scaler, the LinearSVC squared-hinge versus libsvm pilot distinction, and the convergence-only numerical fallback that preserves the original solver budget.
10. Explain all four scientific failures from actual completed run IDs. Two are substantive descriptor/whole-approach failures; two are learning-rate attempts. Engineering interruptions and synthetic checks do not count.
11. State actual final ANN F2 0.6883 versus classical 0.0623 and VLM 0.1391. Explain the pre-evaluation conditional ANN expectation honestly. No test score changed selection.
12. Interpret learning curves and seed spread. Weighted loss gaps also reflect prevalence differences. Two held-out origins make bootstrap intervals coarse; patches are not independent factory samples.
13. Launch the app and upload data/demo cases. Explain the checkpoint hash, fresh request-time prediction, uncalibrated score, frozen class cutoff, separate recommendation costs and limitations. Discuss normal, invalid, blank and noise checks; reliable OOD detection is not implemented.
14. Explain Grad-CAM's defect-logit target and chosen convolutional layer. Compare Integrated Gradients and LIME. Check top/low/random 10% mean-fill occlusion: negative drops mean increased defect score. A zero CAM has no validated hotspot. Map overlap and attractive overlays are not causal proof.
15. Show three genuine errors with original, mask and overlay, their actual scores/cutoff and measured perturbations. Do not invent a manufacturing cause or severity.
16. Explain pinned SmolVLM revision, bfloat16/256 pixels/16 visual tokens/no splitting/greedy 24-token decoding/one retry. Labels, masks, filenames and ANN outputs do not reach inference. Ambiguous responses remain unresolved in all denominators.
17. Discuss local compute/download cost and privacy, label-format failures, compressed tiny detail, possible unknown pretrained-data overlap and a bounded inspection-note extension.
18. Declare the ChatGPT/Codex assistance and scripted real browser video. Review code/results personally and give the individual live presentation under institutional requirements.

Useful evidence: README.md; docs/requirements-matrix.md; reports/final_report.pdf; reports/experiment-log.csv; artifacts/final-evaluation-manifest.json; reports/final-evaluation/; docs/three-xai-errors.json; reports/application-verification/results.json.
