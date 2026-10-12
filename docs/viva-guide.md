# Viva preparation and code understanding

Explain the exact code path: `app/main.py` → `Predictor` → `decode_upload` → shared `preprocess` → selected PyTorch checkpoint → sigmoid/optional temperature → validation cutoffs. The explanation route uses the same model and targets its defect logit. Explain how artifact hashes and strict state loading prevent a silent model swap.

1. **Why this task and user?** State what the inspector does differently and why a patch cannot certify a roll.
2. **Which images are authentic?** Distinguish approved RAW-FABRID from the artificial development sample; do not attribute fictional manufacturer/roll fields to the publisher.
3. **What is a positive label?** Any annotated defect pixel; a normal patch can originate from a defective parent. Explain why unexplained missing masks are rejected.
4. **Why group before splitting?** Neighbouring patches share capture/texture information. Explain source components, duplicate checks, unknown physical links and the new supervised protocol.
5. **Why F2?** Write `F2 = 5TP/(5TP + FP + 4FN)` and explain the decision assumption and always-defective reference. Precision/recall are not confidence.
6. **Explain every CNN layer.** Show tensor sizes and the actual 23,473-parameter initial model. Explain capacity, BatchNorm, pooling and tiny-defect dilution by global averaging.
7. **Why BCEWithLogitsLoss?** One logit for binary labels, stable combined sigmoid/BCE, training-only class weighting and a separately selected decision threshold.
8. **What was controlled?** Identify actual run IDs for learning rate, batch size, true network depth, AdamW/SGD and the two failed approaches. Do not explain an unexecuted plan as an experiment.
9. **Is the classical baseline fair?** Show 24 configurations, train-only scaling, local LBP/HOG features and the same split/metric/threshold policy.
10. **How did you avoid using test during development?** Walk through validation-only scripts, the frozen manifest and exclusive held-out guard. Explain what to do with a partial failure.
11. **What does Grad-CAM establish?** Spatial influence on a logit, not a certified defect outline. Explain empty/zero maps, mask references and why matched-area occlusion is only a sensitivity test.
12. **Explain an incorrect prediction.** Use an actual error, show original/label/overlay, and interpret measured top/low/random drops. Negative drop means occlusion increased the score.
13. **What did the real VLM see?** Pixels plus prompt only, with no label/mask/ANN hint. Explain the fixed version, prompt validation, parsing failures and retained denominators.
14. **When would the baseline or VLM win?** Answer using actual evidence and acquisition/compute limitations, not architecture popularity.
15. **Which work did AI assist?** Declare it separately from Q6 and explain what you personally checked and can reproduce.

For a small-change demonstration, add a new training-only custom-CNN dropout configuration in `search_configs()`, explain its hypothesis, and run it under a new run ID. Do this before final freeze. Alternatively, change the frontend opacity display without changing model decisions. Never change a threshold to improve final test results.

Student action: run the project locally, understand each listed path, reproduce the approved experiments, inspect all claims/citations, and practise explaining both successful and unsuccessful cases. These tasks have not been completed on the student's behalf by creating this guide.
